# -*- coding: utf-8 -*-
import os
import json
import re
import time
from datetime import datetime, timezone
import feedparser
import jieba
import jieba.analyse
import jieba.posseg as pseg
from datetime import timedelta
from deep_translator import GoogleTranslator
from playwright.sync_api import sync_playwright

# ==================== 配置区 ====================
RSS_SOURCES = [
    {
        "source_name": "OpenAI Blog",
        "url": "https://openai.com/news/rss.xml",
        "region": "北美洲"
    },
    {
        "source_name": "Google DeepMind",
        "url": "https://deepmind.google/blog/rss.xml",
        "region": "全球"
    },

    # --- AI 平台聚合商与 API 网关基础设施 ---
    {
        "source_name": "OpenRouter (API聚合商)",
        "url": "https://openrouter.ai/feed.xml",
        "region": "全球"
    },

    # --- 国内高质量专业媒体 ---
    {
        "source_name": "量子位",
        "url": "https://www.qbitai.com/feed",
        "region": "国内"
    },
    {
        "source_name": "机器之心",
        "url": "https://www.jiqizhixin.com/rss",
        "region": "国内"
    }
]

DATA_FILE = "data.json"
HISTORY_DIR = "history"

# ----------------------------------------------------
# 规则词库与清洗正则
# ----------------------------------------------------
TITLE_NOISE_PATTERNS = [
    r'【.*?】', r'\[.*?\]', r'（.*?）', r'\(.*?\)',
    r'[\-_\|].*?(36氪|IT之家|机器之心|OpenAI|腾讯|网易|新浪|搜狐|澎湃|TechCrunch|VentureBeat|AIbase|资讯|快讯).*$'
]

# 相对时间与标记噪声（用于剔除“刚刚”等无效标题）
TIME_NOISE_REGEX = r'^(刚刚|\d+\s*(秒|分钟|小时|天)前|热门|精选|快讯|置顶)$'

# 摘要中需清除的话语标记语
DISCOURSE_MARKERS = [
    r'^据[^\s,，]+(报道|消息)[，,]?', r'^据悉[，,]?', r'^据了解[，,]?', r'^今日[，,]?',
    r'^消息称[，,]?', r'^值得注意的是[，,]?', r'^此外[，,]?', r'^另外[，,]?', r'^根据[^\s,，]+[，,]?'
]

CATEGORY_L1_RULES = {
    "政策": ["政策", "法案", "标准", "评估", "规范", "安全", "治理", "监管", "合规", "指南", "框架", "法规", "policy", "governance"],
    "技术趋势": ["算法", "架构", "推理", "agent", "基准", "评测", "多模态", "突破", "论文", "模型", "芯片", "算力", "微调", "llm"],
    "厂商": ["openai", "google", "微软", "百度", "阿里", "腾讯", "发布", "更新", "推出", "融资", "上线", "收购", "字节", "华为"]
}

CATEGORY_L2_RULES = {
    "开源": ["开源", "github", "huggingface", "weights", "权重", "开放", "apache", "llama", "open-source"],
    "运营商": ["移动", "电信", "联通", "云", "运营商", "专线", "moma", "星辰", "telecom"],
    "聚合": ["聚合", "路由", "中转", "gateway", "网关", "openrouter", "portkey", "litellm", "api"]
}

HIGH_IMPACT_WORDS = ["发布", "突破", "重磅", "安全", "标准", "首个", "开源", "更新", "上线", "推出", "launch", "release"]
TOP_SOURCES = ["OpenAI Blog", "Google Cloud", "36氪", "TechCrunch", "通信产业网", "IT之家", "机器之心", "AIBase"]

# ==================== 辅助工具函数 ====================

def _clean_html(raw_html: str) -> str:
    """去除 HTML 标签及实体符号，保留纯文本"""
    if not raw_html:
        return ""
    text = re.sub(r'<[^>]+>', '', raw_html)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _auto_translate_to_zh(text: str) -> str:
    """检测英文并翻译为中文"""
    if not text:
        return ""
    if len(re.findall(r'[a-zA-Z]{3,}', text)) > 2:
        try:
            translated = GoogleTranslator(source='auto', target='zh-CN').translate(text)
            return translated if translated else text
        except Exception:
            return text
    return text


def _clean_sentence(text: str) -> str:
    """去除句子开头的话语前缀"""
    for pattern in DISCOURSE_MARKERS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
    return text

# ==================== 数据抓取模块 ====================
def _is_today_or_yesterday(pub_dt):
    """校验时间是否为今天或昨天（昨天 00:00:00 之后）"""
    if not pub_dt:
        return True
    now = datetime.now()
    yesterday_start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return pub_dt >= yesterday_start


def _parse_relative_time(time_str):
    """将网页相对时间（如：刚刚、10分钟前、昨天 14:20）解析为 datetime 对象"""
    now = datetime.now()
    time_str = time_str.strip()

    if "刚刚" in time_str:
        return now

    match_min = re.search(r"(\d+)\s*分钟前", time_str)
    if match_min:
        return now - timedelta(minutes=int(match_min.group(1)))

    match_hour = re.search(r"(\d+)\s*小时前", time_str)
    if match_hour:
        return now - timedelta(hours=int(match_hour.group(1)))

    if "昨天" in time_str:
        yesterday = now - timedelta(days=1)
        time_match = re.search(r"(\d{1,2}):(\d{2})", time_str)
        if time_match:
            return yesterday.replace(hour=int(time_match.group(1)), minute=int(time_match.group(2)), second=0)
        return yesterday

    match_date = re.search(r"(\d{4}-)?(\d{2})-(\d{2})", time_str)
    if match_date:
        year = int(match_date.group(1).rstrip('-')) if match_date.group(1) else now.year
        month = int(match_date.group(2))
        day = int(match_date.group(3))
        try:
            return datetime(year, month, day)
        except ValueError:
            pass

    return now


def load_existing_urls(data_file):
    """加载本地已存在的文章链接，避免重复处理"""
    if not os.path.exists(data_file):
        return set()
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {item['url'] for item in data.get('news', []) if 'url' in item}
    except Exception as e:
        print(f"⚠️ 读取已有数据文件失败: {e}")
        return set()


def fetch_rss_items(existing_urls, max_per_feed=3):
    """抓取 RSS 源并提取标准化数据（仅保留今天/昨天的未重复资讯）"""
    new_items = []
    seen_urls = set(existing_urls)

    for source in RSS_SOURCES:
        print(f"📡 正在抓取订阅源: {source['source_name']}...")
        try:
            feed = feedparser.parse(source['url'])
            count = 0
            for entry in feed.entries:
                if count >= max_per_feed:
                    break

                link = entry.get('link', '')
                if not link or link in seen_urls:
                    continue

                # 解析 RSS 发布时间
                pub_dt = datetime.now()
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_dt = datetime(*entry.published_parsed[:6])

                # 过滤掉早于昨天的旧资讯
                if not _is_today_or_yesterday(pub_dt):
                    continue

                title = _clean_html(entry.get('title', ''))
                raw_summary = entry.get('summary', entry.get('description', ''))
                summary = _clean_html(raw_summary)

                new_items.append({
                    "source_name": source['source_name'],
                    "region": source['region'],
                    "raw_title": title,
                    "raw_summary": summary[:1000],
                    "url": link,
                    "pub_time": pub_dt.strftime("%Y-%m-%d %H:%M:%S")
                })
                seen_urls.add(link)
                count += 1
        except Exception as e:
            print(f"❌ 抓取失败 {source['source_name']}: {e}")

    return new_items


def fetch_aibase_news(existing_urls):
    """使用 Playwright 抓取 AIBase（查重 + 仅限今天/昨天的资讯）"""
    url = "https://www.aibase.com/zh/news"
    raw_items = []
    seen_urls = set(existing_urls)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={'width': 1280, 'height': 800}
        )
        page = context.new_page()

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(3000)

            links = page.locator("a[href*='/zh/news/']").all()

            for link in links:
                raw_text = link.inner_text().strip()
                href = link.get_attribute("href") or ""

                if not raw_text:
                    continue

                full_url = href if href.startswith("http") else f"https://www.aibase.com{href}"
                if full_url in seen_urls:
                    continue

                text_lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

                # 提取相对时间文本并计算出真实的 pub_dt
                time_line = next((l for l in text_lines if re.match(TIME_NOISE_REGEX, l)), "")
                pub_dt = _parse_relative_time(time_line) if time_line else datetime.now()

                # 判定是否属于今天或昨天
                if not _is_today_or_yesterday(pub_dt):
                    continue

                valid_lines = [l for l in text_lines if not re.match(TIME_NOISE_REGEX, l)]
                if not valid_lines:
                    continue

                real_title = valid_lines[0]
                if len(real_title) < 5:
                    continue

                seen_urls.add(full_url)
                raw_items.append({
                    "source_name": "AIBase",
                    "raw_title": real_title,
                    "raw_summary": " ".join(valid_lines),
                    "region": "国内",
                    "url": full_url,
                    "pub_time": pub_dt.strftime("%Y-%m-%d %H:%M:%S")
                })

            print(f"✅ [Playwright] 成功抓取 AIBase 资讯 {len(raw_items)} 条。")

        except Exception as e:
            print(f"❌ [Playwright] 抓取失败: {e}")
        finally:
            browser.close()

    return raw_items
# ==================== NLP 高精提炼模块 ====================

def process_without_ai(raw_item: dict) -> dict:
    """本地 NLP 处理：防噪标题、核心句加权摘要、标签与评分计算"""
    raw_title = raw_item.get('raw_title', '').strip()
    raw_summary = raw_item.get('raw_summary', '').strip() or raw_title
    source_name = raw_item.get('source_name', '未知来源')

    # 1. 自动翻译
    zh_title_src = _auto_translate_to_zh(raw_title)
    zh_summary_src = _auto_translate_to_zh(raw_summary)

    # 2. 标题精炼与防噪校验
    clean_title = zh_title_src
    for pattern in TITLE_NOISE_PATTERNS:
        clean_title = re.sub(pattern, '', clean_title, flags=re.IGNORECASE).strip()

    # 安全拦截：若标题仅为时间词或过短，则作废重构
    if re.match(TIME_NOISE_REGEX, clean_title) or len(clean_title) < 4:
        clean_title = zh_summary_src

    # 优雅截断标题（避免断句破坏语法结构）
    if len(clean_title) > 28:
        clauses = [c.strip() for c in re.split(r'[:：,，—–\-]', clean_title) if len(c.strip()) >= 5]
        title_zh = clauses[0] if clauses else clean_title[:26] + "..."
    else:
        title_zh = clean_title

    # 3. 基于句法权重的精确概括
    full_text = f"{zh_title_src}。{zh_summary_src}"
    sentences_raw = [s.strip() for s in re.split(r'[。！？\n;；]', full_text) if len(s.strip()) > 6]
    valid_sentences = []

    for s in sentences_raw:
        if not re.search(r'(点击|关注|来源|图片|未经授权|责任编辑|微信|公众号|版权所有)', s):
            clean_s = _clean_sentence(s)
            if clean_s and not re.match(TIME_NOISE_REGEX, clean_s):
                valid_sentences.append(clean_s)

    if not valid_sentences:
        valid_sentences = [title_zh]

    # 若标题被判定无效，用第一句有效句子补充为标题
    if title_zh == zh_summary_src and valid_sentences:
        title_zh = valid_sentences[0][:26] + ("..." if len(valid_sentences[0]) > 26 else "")

    # 句子评分提取机制
    top_keywords = jieba.analyse.extract_tags(full_text, topK=6)
    scored_sentences = []

    for idx, sentence in enumerate(valid_sentences):
        score = 0.0
        if idx == 0:
            score += 3.0
        elif idx == 1:
            score += 1.5

        words_with_pos = pseg.cut(sentence)
        for word, flag in words_with_pos:
            if flag in ['n', 'nr', 'ns', 'nt', 'nz', 'eng']:
                score += 0.8
            elif flag.startswith('v'):
                score += 0.5

        for kw in top_keywords:
            if kw in sentence:
                score += 1.2

        if 15 <= len(sentence) <= 45:
            score += 1.0

        scored_sentences.append((score, idx, sentence))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    selected_items = sorted(scored_sentences[:2], key=lambda x: x[1])

    summary_body = "；".join([item[2] for item in selected_items])
    if len(summary_body) > 100:
        summary_body = summary_body[:97] + "..."

    # 4. 标签生成
    tags = jieba.analyse.textrank(full_text, topK=3)
    if not tags or len(tags) < 3:
        extra_tags = [kw for kw in top_keywords if kw not in tags]
        tags = (tags + extra_tags + ["AI", "科技", "资讯"])[:3]

    # 5. 分类匹配
    category_l1 = "行业"
    for cat, kws in CATEGORY_L1_RULES.items():
        if any(kw.lower() in full_text.lower() for kw in kws):
            category_l1 = cat
            break

    category_l2 = "原厂"
    for cat, kws in CATEGORY_L2_RULES.items():
        if any(kw.lower() in full_text.lower() for kw in kws):
            category_l2 = cat
            break

    # 6. 重要度打分
    score = 5.0
    if any(top_src.lower() in source_name.lower() for top_src in TOP_SOURCES):
        score += 1.5
    hit_count = sum(1 for word in HIGH_IMPACT_WORDS if word in raw_title.lower())
    score += min(hit_count * 0.8, 2.0)
    if len(raw_summary) > 80:
        score += 1.0

    return {
        "title_zh": title_zh,
        "summary_zh": summary_body,
        "category": {
            "l1": category_l1,
            "l2": category_l2
        },
        "tags": tags,
        "importance_score": round(min(max(score, 1.0), 10.0), 1),
        "source": source_name,
        "region": raw_item.get('region', '中国'),
        "url": raw_item.get('url', '#'),
        "pub_date": raw_item.get('pub_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "created_at": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    }

# ==================== 数据持久化与主流程 ====================

def process_data(raw_items_list):
    return [process_without_ai(item) for item in raw_items_list]


def save_data(new_processed_news):
    if not new_processed_news:
        print("ℹ️ 没有新资讯需要保存。")
        return

    existing_data = {"last_updated": "", "news": []}
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except Exception:
            pass

    updated_news = new_processed_news + existing_data.get("news", [])
    current_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")

    final_payload = {
        "last_updated": current_time,
        "total_count": len(updated_news),
        "news": updated_news
    }

    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)
    print(f"✅ 已成功更新 {DATA_FILE}，当前总计 {len(updated_news)} 条数据。")

    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)
    date_str = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
    history_file = os.path.join(HISTORY_DIR, f"{date_str}.json")
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                today_existing_data = json.load(f)
        except Exception:
            pass
            
    today_updated_news = new_processed_news + today_existing_data.get("news", [])
    today_payload = {
        "last_updated": current_time,
        "total_count": len(today_updated_news),
        "news": today_updated_news
    }
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(today_payload, f, ensure_ascii=False, indent=2)
    print(f"📁 已备份历史快照至 {history_file}")


def main():
    print("🚀 开始运行 AI 资讯自动收集与提炼流水线...\n")

    existing_urls = load_existing_urls(DATA_FILE)
    print(f"🔍 已过滤 {len(existing_urls)} 条历史已收录文章。")

    raw_items = fetch_rss_items(existing_urls, max_per_feed=3)
    aibase_items = fetch_aibase_news(existing_urls)
    raw_items.extend(aibase_items)

    print(f"💡 发现 {len(raw_items)} 条待处理的新资讯。\n")
    save_data(process_data(raw_items))

    print("\n🎉 流水线执行完毕！")


if __name__ == "__main__":
    main()
