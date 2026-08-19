# -*- coding: utf-8 -*-
import os
import json
import time
from datetime import datetime
import feedparser
from openai import OpenAI
import re
import jieba
import jieba.analyse
import jieba.posseg as pseg
from deep_translator import GoogleTranslator
from playwright.sync_api import sync_playwright

# ==================== 配置区 ====================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-074b2023166045b4b945a7a142f8c922")

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
        "url": "https://rsshub.app/github/trending/daily/python", # 示例 RSSHub 源
        "region": "全球"
    }
]

DATA_FILE = "data.json"
HISTORY_DIR = "history"
# ================================================


def get_openai_client():
    """初始化 DeepSeek API 客户端 (兼容 OpenAI SDK)"""
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )


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
            page.wait_for_timeout(3000) # 等待异步数据加载完成
            
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
                        "raw_title": title.split('\n')[0], # 清理多余换行
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

def process_with_deepseek(client, raw_item: dict) -> dict:
    """调用 DeepSeek API 进行归类、摘要与打分（输出结构与 process_without_ai 完全对齐）"""
    system_prompt = """
    你是一个专业的 AI 科技媒体资深编辑与行业分析师。
    请对提供的原始科技资讯进行翻译（若为英文）、提炼与分类。

    请严格返回一个标准的 JSON 对象，格式规范如下（不要包含 markdown 的 ```json 标记）：
    {
      "title_zh": "中文精炼标题（20字以内，去除‘今日/官方/正式’等无意义修饰词与媒体后缀）",
      "summary_zh": "准确概括文章核心大意的中文摘要（100字以内，表达流畅通顺，不包含无意义废话）",
      "category_l1": "一级分类（只能选其一：政策 / 行业 / 技术趋势 / 厂商）",
      "category_l2": "二级分类（只能选其一：原厂 / 聚合 / 运营商 / 开源）",
      "tags": ["中文标签1", "中文标签2", "中文标签3"],
      "importance_score": 8.5
    }
    """

    source_name = raw_item.get('source_name', '未知来源')
    raw_title = raw_item.get('raw_title', '')
    raw_summary = raw_item.get('raw_summary', '') or raw_title

    user_prompt = f"""
    来源：{source_name}
    标题：{raw_title}
    内容：{raw_summary}
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},  # 强制 JSON 输出
            temperature=0.3
        )
        
        result_str = response.choices[0].message.content.strip()
        parsed_data = json.loads(result_str)
        
        # 组装输出结构，与 process_without_ai 的字段与嵌套层次完全一致
        processed_item = {
            "title_zh": parsed_data.get("title_zh", raw_title[:20]),
            "summary_zh": parsed_data.get("summary_zh", raw_summary[:100]),
            "category": {
                "l1": parsed_data.get("category_l1", "行业"),
                "l2": parsed_data.get("category_l2", "原厂")
            },
            "tags": parsed_data.get("tags", ["AI", "科技", "资讯"])[:3],
            "importance_score": float(parsed_data.get("importance_score", 5.0)),
            "source": source_name,
            "url": raw_item.get('url', '#'),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return processed_item

    except Exception as e:
        print(f"❌ AI 处理失败 ({raw_title[:15]}...): {e}")
        # 异常发生时，返回符合同一结构的降级数据，确保数据流中断
        return {
            "title_zh": raw_title[:20],
            "summary_zh": raw_summary[:100],
            "category": {
                "l1": "行业",
                "l2": "原厂"
            },
            "tags": ["AI", "科技", "资讯"],
            "importance_score": 5.0,
            "source": source_name,
            "url": raw_item.get('url', '#'),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

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
    
    client = get_openai_client()
    existing_urls = load_existing_urls(DATA_FILE)
    print(f"🔍 已过滤 {len(existing_urls)} 条历史已收录文章。")

    # 抓取未处理的 RSS
    raw_items = fetch_rss_items(existing_urls, max_per_feed=3)
    aibase_items = fetch_aibase_news()
    raw_items.extend(aibase_items)
    print(f"💡 发现 {len(raw_items)} 条待处理的新资讯。\n")
    
    processed_news = []
    for item in raw_items:
        print(f"🤖 正在调用 DeepSeek 处理: {item['raw_title'][:30]}...")
        result = process_with_deepseek(client, item)
        if result:
            processed_news.append(result)
        time.sleep(0.5)  # 轻微延迟，避免频繁触发 API 限流

    #保存文件
    save_data(processed_news)
    
    #no ai
    #save_data(process_data(raw_items))

    print("\n🎉 流水线执行完毕！")

if __name__ == "__main__":
    main()