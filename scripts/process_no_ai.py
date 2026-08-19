# -*- coding: utf-8 -*-
import os
import json
from datetime import datetime
import feedparser
import re
import jieba
import jieba.analyse
import jieba.posseg as pseg
from deep_translator import GoogleTranslator
from playwright.sync_api import sync_playwright

# ==================== 配置区 ====================
# RSS 订阅源配置列表
RSS_SOURCES = [
    {
        "source_name": "OpenAI Blog",
        "url": "https://openai.com/news/rss.xml",
        "region": "北美洲"
    },
    {
        "source_name": "MIT Tech Review AI",
        "url": "https://www.technologyreview.com/topic/artificial-intelligence/feed/",
        "region": "全球"
    },
    {
        "source_name": "GitHub Trending AI",
        "url": "https://rsshub.app/github/trending/daily/python",  # 示例 RSSHub 源
        "region": "全球"
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

# 标题中需强力剔除的填充虚词（压缩字数）
TITLE_FILLER_WORDS = ["今日", "官方", "正式", "最新", "宣布", "针对", "关于", "进行了", "发布了", "推出了", "提出了",
                      "重磅", "全面"]

# 摘要中需清除的话语标记语（提升文本严肃度）
DISCOURSE_MARKERS = [
    r'^据[^\s,，]+(报道|消息)[，,]?', r'^据悉[，,]?', r'^据了解[，,]?', r'^今日[，,]?',
    r'^消息称[，,]?', r'^值得注意的是[，,]?', r'^此外[，,]?', r'^另外[，,]?', r'^根据[^\s,，]+[，,]?'
]

CATEGORY_L1_RULES = {
    "政策": ["政策", "法案", "标准", "评估", "规范", "安全", "治理", "监管", "合规", "指南", "框架", "法规", "policy",
             "governance"],
    "技术趋势": ["算法", "架构", "推理", "agent", "基准", "评测", "多模态", "突破", "论文", "模型", "芯片", "算力",
                 "微调", "llm"],
    "厂商": ["openai", "google", "微软", "百度", "阿里", "腾讯", "发布", "更新", "推出", "融资", "上线", "收购", "字节",
             "华为"]
}

CATEGORY_L2_RULES = {
    "开源": ["开源", "github", "huggingface", "weights", "权重", "开放", "apache", "llama", "open-source"],
    "运营商": ["移动", "电信", "联通", "云", "运营商", "专线", "moma", "星辰", "telecom"],
    "聚合": ["聚合", "路由", "中转", "gateway", "网关", "openrouter", "portkey", "litellm", "api"]
}

HIGH_IMPACT_WORDS = ["发布", "突破", "重磅", "安全", "标准", "首个", "开源", "更新", "上线", "推出", "launch",
                     "release"]
TOP_SOURCES = ["OpenAI Blog", "Google Cloud", "36氪", "TechCrunch", "通信产业网", "IT之家", "机器之心"]
# ================================================
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
    """抓取 RSS 源并进行初步去重"""
    new_items = []

    for source in RSS_SOURCES:
        print(f"📡 正在抓取订阅源: {source['source_name']}...")
        try:
            feed = feedparser.parse(source['url'])
            count = 0
            for entry in feed.entries:
                if count >= max_per_feed:
                    break

                link = entry.get('link', '')
                if not link or link in existing_urls:
                    continue  # 跳过已有文章

                title = entry.get('title', '')
                summary = entry.get('summary', entry.get('description', ''))
                pub_time = entry.get('published', datetime.now().strftime("%Y-%m-%d %H:%M"))

                new_items.append({
                    "source_name": source['source_name'],
                    "region": source['region'],
                    "raw_title": title,
                    "raw_summary": summary[:1000],  # 截断避免 Token 过长
                    "url": link,
                    "pub_time": pub_time
                })
                count += 1
        except Exception as e:
            print(f"❌ 抓取失败 {source['source_name']}: {e}")

    return new_items


def fetch_aibase_news():
    """使用 Playwright 渲染页面并提取数据"""
    url = "https://www.aibase.com/zh/news"
    raw_items = []

    with sync_playwright() as p:
        # 启动无头 Chrome，设置真实浏览器特征
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={'width': 1280, 'height': 800}
        )
        page = context.new_page()

        try:
            # 访问页面并等待 DOM 内容加载完成
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(3000)  # 等待异步数据加载完成

            # 抓取新闻链接元素
            links = page.locator("a[href*='/zh/news/']").all()
            seen_urls = set()

            for link in links:
                title = link.inner_text().strip()
                href = link.get_attribute("href") or ""

                if not title or len(title) < 6:
                    continue

                full_url = href if href.startswith("http") else f"https://www.aibase.com{href}"

                if full_url not in seen_urls:
                    seen_urls.add(full_url)
                    raw_items.append({
                        "source_name": "AIBase",
                        "raw_title": title.split('\n')[0],  # 清理多余换行
                        "raw_summary": title.replace('\n', ' '),
                        "region": "国内",
                        "url": full_url
                    })

            print(f"✅ [Playwright] 成功抓取 AIBase 资讯 {len(raw_items)} 条。")

        except Exception as e:
            print(f"❌ [Playwright] 抓取失败: {e}")
        finally:
            browser.close()

    return raw_items

def _auto_translate_to_zh(text: str) -> str:
    """自动检测英文并翻为中文"""
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


def process_without_ai(raw_item: dict) -> dict:
    """
    高精度本地 NLP 处理：核心词性抽词、标题修饰词剪裁、关键句概率重构
    """
    raw_title = raw_item.get('raw_title', '').strip()
    raw_summary = raw_item.get('raw_summary', '').strip() or raw_title
    source_name = raw_item.get('source_name', '未知来源')

    # 1. 中文自动转换
    zh_title_src = _auto_translate_to_zh(raw_title)
    zh_summary_src = _auto_translate_to_zh(raw_summary)

    # --------------------------------------------------
    # 2. 标题精炼剪裁（去除噪声 + 过滤无实义修饰词）
    # --------------------------------------------------
    clean_title = zh_title_src
    for pattern in TITLE_NOISE_PATTERNS:
        clean_title = re.sub(pattern, '', clean_title, flags=re.IGNORECASE).strip()

    # 尝试剔除填充虚词以压缩字数
    short_title = clean_title
    if len(short_title) > 20:
        for fw in TITLE_FILLER_WORDS:
            if len(short_title) > 20:
                short_title = short_title.replace(fw, "")

    # 若字数仍超标，按标点切分取核心子句
    if len(short_title) > 20:
        clauses = [c.strip() for c in re.split(r'[:：,，—–-]', short_title) if len(c.strip()) >= 5]
        valid_clauses = [c for c in clauses if len(c) <= 20]
        title_zh = valid_clauses[0] if valid_clauses else short_title[:18] + "..."
    else:
        title_zh = short_title if short_title else clean_title[:20]

    # --------------------------------------------------
    # 3. 基于“实体+动词”密度的精确概括算法
    # --------------------------------------------------
    full_text = f"{zh_title_src}。{zh_summary_src}"

    # 切分句子并清洗前缀
    sentences_raw = [s.strip() for s in re.split(r'[。！？\n;；]', full_text) if len(s.strip()) > 6]
    valid_sentences = []
    for s in sentences_raw:
        if not re.search(r'(点击|关注|来源|图片|未经授权|责任编辑|微信|公众号|版权所有)', s):
            clean_s = _clean_sentence(s)
            if clean_s:
                valid_sentences.append(clean_s)

    if not valid_sentences:
        valid_sentences = [title_zh]

    # 提取全局核心关键词
    top_keywords = jieba.analyse.extract_tags(full_text, topK=6)

    # 计算句子“信息量权重”
    scored_sentences = []
    for idx, sentence in enumerate(valid_sentences):
        score = 0.0

        # A. 句子首尾位置得分（导语句信息量最高）
        if idx == 0:
            score += 3.0
        elif idx == 1:
            score += 1.5

        # B. 词性标注打分（名词/专有名词/动词越多，信息密度越高）
        words_with_pos = pseg.cut(sentence)
        for word, flag in words_with_pos:
            if flag in ['n', 'nr', 'ns', 'nt', 'nz', 'eng']:  # 名词、机构名、英文缩写
                score += 0.8
            elif flag.startswith('v'):  # 核心动词
                score += 0.5

        # C. 命中核心关键词加分
        for kw in top_keywords:
            if kw in sentence:
                score += 1.2

        # D. 理想长度区间加分 (15 - 45 字)
        if 15 <= len(sentence) <= 45:
            score += 1.0

        scored_sentences.append((score, idx, sentence))

    # 挑选信息密度最高的前 1-2 句组合成最终摘要
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    selected_items = sorted(scored_sentences[:2], key=lambda x: x[1])

    summary_body = "；".join([item[2] for item in selected_items])
    if len(summary_body) > 100:
        summary_body = summary_body[:97] + "..."

    summary_zh = summary_body

    # --------------------------------------------------
    # 4. 提取 3 个高质量中文标签
    # --------------------------------------------------
    tags = jieba.analyse.textrank(full_text, topK=3)
    if not tags or len(tags) < 3:
        extra_tags = [kw for kw in top_keywords if kw not in tags]
        tags = (tags + extra_tags + ["AI", "科技", "资讯"])[:3]

    # --------------------------------------------------
    # 5. 一级与二级分类匹配
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 6. 启发式加权打分 (1.0 - 10.0)
    # --------------------------------------------------
    score = 5.0
    if any(top_src.lower() in source_name.lower() for top_src in TOP_SOURCES):
        score += 1.5
    hit_count = sum(1 for word in HIGH_IMPACT_WORDS if word in raw_title.lower())
    score += min(hit_count * 0.8, 2.0)
    if len(raw_summary) > 80:
        score += 1.0
    importance_score = round(min(max(score, 1.0), 10.0), 1)

    # --------------------------------------------------
    # 7. 返回标准数据结构
    # --------------------------------------------------
    return {
        "title_zh": title_zh,
        "summary_zh": summary_zh,
        "category": {
            "l1": category_l1,
            "l2": category_l2
        },
        "tags": tags,
        "importance_score": importance_score,
        "source": source_name,
        "region": raw_item.get('region', '中国'),
        "url": raw_item.get('url', '#'),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def process_data(raw_items_list):
    """
    批量处理抓取到的原始列表数据，并自动写出为本地 data.json 文件
    """
    processed_news = [process_without_ai(item) for item in raw_items_list]

    return processed_news

def save_data(new_processed_news):
    """保存更新后的数据至主文件及历史归档目录"""
    if not new_processed_news:
        print("ℹ️ 没有新资讯需要保存。")
        return

    # 1. 读取历史数据
    existing_data = {"last_updated": "", "news": []}
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except Exception:
            pass

    # 2. 合并并更新主 JSON
    updated_news = new_processed_news + existing_data.get("news", [])
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    final_payload = {
        "last_updated": current_time,
        "total_count": len(updated_news),
        "news": updated_news
    }

    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)
    print(f"✅ 已成功更新 {DATA_FILE}，当前总计 {len(updated_news)} 条数据。")

    # 3. 按日备份快照至 history/ 目录
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)

    date_str = datetime.now().strftime("%Y-%m-%d")
    history_file = os.path.join(HISTORY_DIR, f"{date_str}.json")

    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)
    print(f"📁 已备份历史快照至 {history_file}")


def main():
    print("🚀 开始运行 AI 资讯自动收集与提炼流水线...\n")

    existing_urls = load_existing_urls(DATA_FILE)
    print(f"🔍 已过滤 {len(existing_urls)} 条历史已收录文章。")

    # 抓取未处理的 RSS
    raw_items = fetch_rss_items(existing_urls, max_per_feed=3)
    aibase_items = fetch_aibase_news()
    raw_items.extend(aibase_items)
    print(f"💡 发现 {len(raw_items)} 条待处理的新资讯。\n")

    save_data(process_data(raw_items))

    print("\n🎉 流水线执行完毕！")


if __name__ == "__main__":
    main()
