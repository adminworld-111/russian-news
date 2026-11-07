# =====================================================================
# ロシア語ニュース学習アプリ　GitHub Actions版
# 作成：2026-11-01（Colab版 2026-10-31 時点のコードをもとに移植）
#
# 実行環境：GitHub Actions（毎朝自動で実行）
# 実行方法：python main.py
# 前提　　：リポジトリのシークレットに ANTHROPIC_API_KEY を登録しておく
#
# Colab版との違い：
#   - ライブラリは requirements.txt でインストールする（!pip は使わない）
#   - APIキーは環境変数から読む（Colabのシークレットは使わない）
#   - 保存先はリポジトリの data/ フォルダ（Googleドライブは使わない）
#   - 記事の本文は保存しない（公開リポジトリのため。教材の生成はメモリ上の本文で行う）
#   - 収集から教材の保存まで、1回の実行で通して行う
#   - 今日の selected.json が既にあれば、何もせずに終わる（1日1回だけ作る）
#
# Colab版は、プロンプトの比較などの実験用にそのまま残す
#
# 構成（Colab版のセルと対応）：
#   1 設定 / 2 AI呼び出し層 / 3 保存 / 4 収集と選定
#   5 教材生成 / 6 教材のチェック / 7 文法問題の検証 / 8 毎日の実行（main）
# =====================================================================

import glob, json, os, re, sys
from datetime import datetime
from zoneinfo import ZoneInfo

import anthropic, feedparser, requests
from bs4 import BeautifulSoup


# %% 1 設定 ===============================================================

# APIキーは環境変数から読む（GitHub Actionsがシークレットを環境変数として渡す）
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# どの仕事にどのAIを使うか（ここを書き換えるだけで切り替え可能）
MODELS = {
    "classify": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "max_tokens": 300},
    "lesson":   {"provider": "anthropic", "model": "claude-sonnet-5",           "max_tokens": 16000},
    "check":    {"provider": "anthropic", "model": "claude-sonnet-5",           "max_tokens": 16000},
}

# ニュースソース（Gorkyは RSS なしのため見送り）
FEEDS = {
    "BBC": "https://feeds.bbci.co.uk/russian/rss.xml",
    "Meduza": "https://meduza.io/rss/news",
    "RBK": "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
    "Interfax": "https://www.interfax.ru/rss.asp",
    "Kommersant": "https://www.kommersant.ru/RSS/news.xml",
    "Kommersant_Culture": "https://www.kommersant.ru/RSS/section-culture.xml",
    # "NPlus1": "https://nplus1.ru/rss",   # 記事ページはCloudflareで拒否、RSSはリード文のみ → 保留
}

# 本文の入れ物となるタグ（サイトの作りが標準と違う媒体だけ指定する）
BODY_AREA = {
    "Kommersant": "main",           # 2026年10月のサイト刷新で、<article>が関連記事の枠になった
    "Kommersant_Culture": "main",
}

# 収集と選定のルール
PER_FEED = 6          # 各媒体から試す記事数
MIN_LENGTH = 500      # これ未満の本文はスキップ
EXCLUDE = {"war"}     # 除外するジャンル
MIN_NEUTRALITY = 4    # 中立度の下限（1〜5）
MIN_VALUE = 3         # 学習価値の下限（1〜5）
PRIORITY = ["culture", "science", "economy", "society", "sports", "incident", "politics"]
DAILY_COUNT = 3       # 1日に作る教材の数

# 保存先
DATA_DIR = "data"            # リポジトリの中の data フォルダ（Colab版はGoogleドライブ）
TIMEZONE = "Asia/Tokyo"      # 日付を決めるタイムゾーン（GitHubのサーバーはUTCなので必ず指定する）


# %% 2 AI呼び出し層 =======================================================

def call_llm(task, system, user):
    """仕事名（task）に応じたAIを呼び、答えの文字列だけを返す"""
    cfg = MODELS[task]
    if cfg["provider"] == "anthropic":
        response = client.messages.create(
            model=cfg["model"], max_tokens=cfg["max_tokens"],
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if response.stop_reason == "max_tokens":
            print(f"⚠ [{task}] 出力が途中で打ち切られました")
        # 思考ブロックなどを除き、答え（text）のブロックだけを取り出す
        return "".join(b.text for b in response.content if b.type == "text")
    # 将来、Geminiなどを追加するときは、ここに分岐を足す
    raise ValueError(f"未対応のプロバイダー：{cfg['provider']}")

def parse_json(text):
    """AIの返答をJSONとして読む。失敗したらエラー箇所を表示してNoneを返す"""
    raw = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"JSONエラー：{e.msg}（{e.pos}文字目）")
        print("問題の箇所：", raw[max(0, e.pos - 100):e.pos + 50])
        return None

def call_llm_json(task, system, user):
    return parse_json(call_llm(task, system, user))


# %% 3 保存 ===============================================================

def today_str():
    """日本時間の今日の日付を「2026-11-01」の形で返す"""
    return datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d")

def save_json(data, filename, date=None):
    """データを日付フォルダにJSONファイルとして保存する"""
    folder = os.path.join(DATA_DIR, date or today_str())   # 例：data/2026-11-01
    os.makedirs(folder, exist_ok=True)                      # フォルダがなければ作る
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)    # キリル文字を読める形で保存
    print(f"💾 保存：{path}")
    return path


# %% 4 収集と選定 =========================================================

def get_feed(name, url):
    """RSSを取得して記事の一覧を返す。失敗したら空のリストを返す"""
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    except requests.RequestException as e:
        print(f"⚠ RSS接続エラー（{type(e).__name__}）：{name}")
        return []
    if res.status_code != 200:
        print(f"⚠ RSS取得失敗（ステータス{res.status_code}）：{name}")
        return []
    return feedparser.parse(res.content).entries     # 文字コードはfeedparserが自動で判断する

def clean_html(html):
    """HTMLタグや&laquo;などを取り除いてプレーンテキストにする"""
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)

def get_body_from_page(url, area_tag=None):
    """記事ページをダウンロードし、本文の段落をつなげる（汎用ルート）"""
    try:
        page = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    except requests.RequestException as e:
        print(f"⚠ 接続エラー（{type(e).__name__}）：{url}")
        return ""
    if page.status_code != 200:                    # 403（拒否）や404（なし）など
        print(f"⚠ 取得失敗（ステータス{page.status_code}）：{url}")
        return ""
    page.encoding = page.apparent_encoding         # 文字化け対策
    soup = BeautifulSoup(page.text, "html.parser")
    if area_tag:                                   # 媒体ごとに指定があれば、そのタグを使う
        area = soup.find(area_tag)
    else:                                          # 指定がなければ、標準の探し方
        area = soup.find("article") or soup.find("main")
    if area is None:
        print(f"⚠ 本文の範囲が見つからない：{url}")
        return ""
    body = " ".join(p.get_text(" ", strip=True) for p in area.find_all("p"))
    if not body:                                   # 範囲は見つかったが、段落が1つもない
        print(f"⚠ 本文の段落が見つからない：{url}")
    return body

def get_body(name, entry):
    """媒体ごとに本文の取り方を切り替える"""
    if name == "RBK":
        return clean_html(entry.get("rbc_news_full-text", ""))
    if name == "Meduza":
        return clean_html(entry.content[0].value) if "content" in entry else ""
    # その他の媒体は、記事ページから本文を取る（汎用ルート）
    return get_body_from_page(entry.link, BODY_AREA.get(name))

CLASSIFY_PROMPT = """あなたはロシア語ニュースの編集者です。
与えられた記事を分類し、次の形式のJSONだけを返してください。前置きや```は不要です。

{"category": "politics / war / economy / society / science / culture / sports / incident のどれか1つ",
 "neutrality": 1〜5の整数,
 "value": 1〜5の整数,
 "russia": 1〜5の整数,
 "reason": "判定理由を日本語で1文"}

categoryの基準：
- war：戦闘・攻撃・軍事支援・停戦交渉・戦争に関わる制裁など、武力紛争が話題の中心にある記事。政治家の発言や外交の記事でも、話題が戦争であればwarにする
- politics：選挙・法律・政府の人事・外交など、武力紛争以外の政治の記事

neutralityの基準：
5 = 事実中心で、評価的・感情的な表現がほぼない
3 = 評価的な表現が一部ある、または一方の見解が中心
1 = 感情的・一方的な主張が中心

valueの基準（上級ロシア語学習者にとっての教材価値）：
5 = 背景や論点があり、読んで考える内容が豊か
3 = 事実報道として標準的
1 = 短い速報、人名の列挙、事務的な告知が中心

russiaの基準（ロシアとの関わり）：
5 = ロシア国内の政治・経済・社会・文化・科学が話題の中心
3 = 他国や国際的な話題だが、ロシアが主要な当事者として関わる
1 = ロシアとほぼ関係がない

媒体名ではなく、本文の書きぶりだけで判断してください。
引用された発言に含まれる評価的表現は、記事自体の評価には含めないでください。
文字列の中で引用符が必要な場合は «» または「」を使い、半角の " は絶対に使わないでください。"""

def classify(article):
    text = f"見出し: {article['title']}\n本文冒頭: {article['body'][:800]}"
    meta = call_llm_json("classify", CLASSIFY_PROMPT, text)
    if meta:                                       # JSONが読めたときだけ整える
        # 空白や大文字の揺れをなくし、ジャンルの重複チェックを確実にする
        meta["category"] = str(meta.get("category", "")).strip().lower()
    return meta

def priority_of(article):
    cat = article["meta"]["category"]
    return PRIORITY.index(cat) if cat in PRIORITY else len(PRIORITY)

def collect_and_select():
    """記事を集めて判定し、教材にする記事を選ぶ。（全記事, 選んだ記事）を返す"""
    # --- 収集 ---
    articles = []
    for name, url in FEEDS.items():
        entries = get_feed(name, url)[:PER_FEED]
        count = 0
        for entry in entries:
            body = get_body(name, entry)
            if len(body) >= MIN_LENGTH:
                articles.append({"source": name, "title": entry.title,
                                 "link": entry.link, "body": body})
                count += 1
        print(f"取得：{name}　RSS {len(entries)}件 → 本文あり {count}件")   # 媒体ごとの結果
    print(f"収集：{len(articles)}件")

    # --- 判定 ---
    for a in articles:
        a["meta"] = classify(a)

    # --- 選定 ---
    candidates = [a for a in articles
                  if a["meta"]
                  and a["meta"]["category"] not in EXCLUDE
                  and a["meta"]["neutrality"] >= MIN_NEUTRALITY
                  and a["meta"].get("value", 0) >= MIN_VALUE]
    candidates.sort(key=lambda a: (-a["meta"].get("russia", 0),   # ① ロシア関連度の高い順
                                   priority_of(a),                 # ② ジャンルの優先順
                                   -a["meta"]["neutrality"]))      # ③ 中立度の高い順

    selected, used = [], set()
    for a in candidates:
        if a["meta"]["category"] in used:          # 既に選んだジャンルなら飛ばす
            continue
        selected.append(a)
        used.add(a["meta"]["category"])
        if len(selected) == DAILY_COUNT:
            break

    print(f"候補 {len(candidates)}件 → 今日の教材 {len(selected)}件")
    for i, a in enumerate(selected):
        m = a["meta"]
        print(f"{i}: [{a['source']}] {m['category']} / 中立度{m['neutrality']} / 価値{m.get('value')} / ロシア{m.get('russia')} - {a['title']}")
    return articles, selected


# %% 5 教材生成 ===========================================================

GENERATE_PROMPT = """あなたはロシア語検定1級レベルの学習者向け教材を作る、経験豊富なロシア語教師です。
与えられたニュース記事をもとに、次の形式のJSONだけを返してください。前置きや```は不要です。

{
  "summary_ru": "記事内容のロシア語要約（200〜300語）。元記事の文をそのまま使わず、自分の言葉で書き直すこと",
  "summary_ja": "summary_ruの日本語訳",
  "comprehension": [
    {"question": "内容理解の問い（ロシア語）", "options": ["選択肢A", "選択肢B", "選択肢C"], "answer": 0, "explanation": "日本語の解説"}
  ],
  "vocabulary": [
    {"word": "アクセント記号付きの単語", "meaning": "日本語の意味", "grammar": "格支配・体のペア・派生語など", "example": "summary_ruからの例文"}
  ],
  "phrases": [
    {"phrase": "熟語・コロケーション", "meaning": "日本語の意味", "example": "summary_ruからの例文"}
  ],
  "grammar": [
    {"question": "空欄（___）を含むロシア語の文", "options": ["選択肢A", "選択肢B", "選択肢C", "選択肢D"], "answer": 0, "explanation": "日本語の文法解説"}
  ],
  "cultural_notes": [
    {"reference": "文学・歴史・文化的な引用や背景知識", "explanation": "日本語の解説"}
  ]
}

作成ルール：
- meaning、explanation（comprehension・grammar・cultural_notesのすべて）、summary_jaは必ず日本語で書く。記事や問題文がロシア語でも、これらの欄をロシア語で書かない
- comprehensionは2問、vocabularyは6〜8語、phrasesは2〜3個、grammarは3問、cultural_notesは0〜3個
- answerは正解の選択肢の番号（0から数える）
- summary_ruはC1〜C2レベルの新聞・論説の文体で書き、形動詞・副動詞構文や複文を適度に含める。人名の羅列は避け、内容の要点を優先する
- vocabularyは1語の単語だけを選ぶ。2語以上のまとまり（熟語・コロケーション）はphrasesに入れる。同じ語句をvocabularyとphrasesの両方に入れない
- vocabularyとphrasesは、必ずsummary_ruに実際に出てくる語から選ぶ。wordは辞書形にしてよいが、summary_ruで使われている語そのものであること。似た綴りの別の語や類義語に置き換えない。国際語や、上級学習者が既に知っている語は避ける
- vocabularyのgrammar欄には、格支配、対応する動詞や体のペア、派生語など、上級学習者に役立つ情報を書く。単なる性や格の表示は不要
- exampleはsummary_ruの文を一字一句そのまま使う。長い文は一部を切り出してよいが、語を足したり変えたりしない。元記事の文は、例文にも解説にも一切使わない
- grammarは格変化、動詞の体、形動詞・副動詞、前置詞の使い分けなど、検定1級で問われる文法事項から出題する
- grammarの選択肢は、同じ語の変化形同士、体のペア、形動詞の種類の違いなど、上級学習者でも迷う組み合わせにする。明らかに品詞や構文が合わない選択肢や、時制・人称・数の不一致、不定詞など、文の形だけで消去できる選択肢は入れない
- grammarの正解は文法的に1つだけに決まること。他の選択肢は、その文脈では文法的に誤りになるものにする。「意味に合わない」「文脈にそぐわない」といった解釈の違いで正解が決まる問題は作らない。解説では、他の選択肢が文法的に誤りである理由を説明する
- cultural_notesには、見出しや本文に含まれる文学作品のもじり、歴史的な出来事、文化的な固有名詞など、背景知識がないと理解しにくいものを選ぶ。該当がなければ空のリストにする。作品の種類、日付、人物の経歴など、記事に書かれている事実と矛盾する内容は書かない
- すべての問題はsummary_ruの内容に基づいて作る
- 文字列の中で引用符が必要な場合は «» または「」を使い、半角の " は絶対に使わない"""

def generate_lesson(article):
    text = f"媒体: {article['source']}\n見出し: {article['title']}\n本文: {article['body'][:6000]}"
    return call_llm_json("lesson", GENERATE_PROMPT, text)


# %% 6 教材のチェック =====================================================
# 形式のルール違反だけを探す。ロシア語の自然さや文法問題の正解の一意性は判定できない

JA = re.compile(r"[ぁ-んァ-ヶ一-龯]")    # ひらがな・カタカナ・漢字のどれか

def plain(text):
    """アクセント記号（\\u0301）を取り除く（要約にはアクセント記号がないため）"""
    return text.replace("\u0301", "")

def norm(text):
    """比較用に整える：アクセント記号を除き、小文字にし、ё を е にそろえる"""
    return plain(text).lower().replace("ё", "е")

def fragments(text):
    """「...」「…」で区切り、前後の句読点を除いた断片のリストを返す"""
    parts = [p.strip(" .,;:") for p in re.split(r"\.\.\.|…", norm(text))]
    return [p for p in parts if p]                 # 空の断片を捨てる

def check_lesson(lesson):
    """教材をルールに照らしてチェックし、問題点のリストを返す"""
    problems = []
    summary = norm(lesson["summary_ru"])           # 例文と同じ方法で整えておく

    # 1. 解説が日本語で書かれているか
    for sec in ["comprehension", "grammar", "cultural_notes"]:
        for i, item in enumerate(lesson.get(sec, [])):
            if not JA.search(item.get("explanation", "")):
                problems.append(f"{sec}[{i}] 解説が日本語でない")

    # 2. 意味が日本語か、例文が要約と一字一句同じか
    for sec in ["vocabulary", "phrases"]:
        for i, item in enumerate(lesson.get(sec, [])):
            if not JA.search(item.get("meaning", "")):
                problems.append(f"{sec}[{i}] 意味が日本語でない")
            frags = fragments(item.get("example", ""))
            if not frags:                          # 例文が空のとき
                problems.append(f"{sec}[{i}] 例文が空")
            elif not all(f in summary for f in frags):   # 断片が1つでも要約になければ違反
                problems.append(f"{sec}[{i}] 例文が要約と一致しない：{item.get('example', '')[:40]}…")

    # 3. vocabularyが1語だけか、phrasesと重複していないか
    phrases = {plain(p["phrase"]) for p in lesson.get("phrases", [])}
    for i, v in enumerate(lesson.get("vocabulary", [])):
        word = plain(v["word"])
        if " " in word:
            problems.append(f"vocabulary[{i}] 1語でない：{word}")
        if word in phrases:
            problems.append(f"vocabulary[{i}] phrasesと重複：{word}")

    # 4. answerの番号が選択肢の範囲内か
    for sec in ["comprehension", "grammar"]:
        for i, q in enumerate(lesson.get(sec, [])):
            if not 0 <= q["answer"] < len(q["options"]):
                problems.append(f"{sec}[{i}] answerの番号が範囲外：{q['answer']}")

    return problems


# %% 7 文法問題の検証 =====================================================
# 正解を伏せたまま、選択肢ごとに文法的に成立するかをAIに判定させ、
# 「成立するのが正解の1つだけか」をプログラムで確かめる。結果は各問題の check 欄に書き込む

CHECK_GRAMMAR_PROMPT = """あなたはロシア語の文法を厳密に判定する校閲者です。
空欄（___）を含むロシア語の文と、番号付きの候補が与えられます。
候補を1つずつ空欄に入れ、その文がロシア語として文法的に成立するかを判定してください。

判定のルール：
- 作問者の意図や、どれが最も自然かは考えない。文法的に成立するかだけを判定する
- 意味や文体がやや不自然でも、文法的に成立するならtrueにする
- 格・性・数の一致、動詞の体と時制、形動詞・副動詞の用法、前置詞と格の支配を確認する
- 判定理由は日本語で1文で書く

次の形式のJSONだけを返してください。前置きや```は不要です。
候補と同じ順番で、すべての候補について返してください。
{"results": [{"index": 0, "grammatical": true, "reason": "判定理由"}]}
文字列の中で引用符が必要な場合は «» または「」を使い、半角の " は絶対に使わないでください。"""

def check_grammar(lesson):
    """文法問題を検証し、結果を各問題の check 欄に書き込む。問題点のリストを返す"""
    problems = []
    for i, q in enumerate(lesson.get("grammar", [])):
        if "check" not in q:                       # まだ検証していない問題だけAIに判定させる
            options_text = "\n".join(f"{n}: {o}" for n, o in enumerate(q["options"]))
            user = f"文：{q['question']}\n候補：\n{options_text}"
            result = call_llm_json("check", CHECK_GRAMMAR_PROMPT, user)
            if not result:
                problems.append(f"grammar[{i}] 判定に失敗した")
                continue
            valid = [r["index"] for r in result["results"] if r["grammatical"]]
            reasons = [""] * len(q["options"])     # 選択肢の数だけ空の理由を用意する
            for r in result["results"]:
                reasons[r["index"]] = r["reason"]  # 番号の位置に理由を入れる
            q["check"] = {"ok": valid == [q["answer"]], "valid": valid, "reasons": reasons}

        c = q["check"]
        if not c["ok"]:
            names = [q["options"][n] for n in c["valid"]]
            print(f"  ⚠ grammar[{i}] 正解が複数ある可能性：{names}")
            problems.append(f"grammar[{i}] 成立する選択肢：{names}（正解：{q['options'][q['answer']]}）")
    return problems


# %% 8 毎日の実行 =========================================================

def public_info(article):
    """公開してよい項目だけを取り出す（本文は除く）"""
    return {"source": article["source"], "title": article["title"],
            "link": article["link"], "meta": article.get("meta")}

def main():
    date = today_str()
    folder = os.path.join(DATA_DIR, date)

    # 今日の分が既にあれば、何もしない（1日1回だけ作る。手動で再実行しても費用がかからない）
    if os.path.exists(os.path.join(folder, "selected.json")):
        print(f"本日分（{date}）は作成済みです。何もせずに終わります")
        return

    # 前回の実行が途中で失敗して残った教材があれば消す（新しい選定と番号がずれるのを防ぐ）
    for path in glob.glob(os.path.join(folder, "lesson_*.json")):
        os.remove(path)

    # 収集・判定・選定。判定した全記事の結果も保存する（本文は除く）
    articles, selected = collect_and_select()
    save_json([public_info(a) for a in articles], "classified.json", date)

    results = []
    for i, article in enumerate(selected):
        print(f"\n{'=' * 60}\n[{i}] {article['title']}")
        try:                                       # 1本で問題が起きても、残りの記事は続ける
            lesson = generate_lesson(article)
            if lesson is None:
                print("→ 教材の生成に失敗したので飛ばします")
                continue
            format_problems = check_lesson(lesson)
            for p in format_problems:
                print("  形式：", p)
            grammar_problems = check_grammar(lesson)
            save_json(lesson, f"lesson_{i}.json", date)
            results.append((i, article["title"], len(format_problems), len(grammar_problems)))
        except Exception as e:
            print(f"→ エラーのため飛ばします（{type(e).__name__}：{e}）")

    # selected.json は最後に保存する（これがあれば「今日の分は完成」という印になる）
    save_json([public_info(a) for a in selected], "selected.json", date)

    print(f"\n{'=' * 60}\n■ まとめ（{date}）")
    for i, title, f_count, g_count in results:
        print(f"[{i}] 形式の問題 {f_count}件 / 正解が複数の可能性 {g_count}問 - {title[:40]}")

    # 教材が1本もできなかったら、失敗として終わる（GitHub Actionsの画面に赤い印が付く）
    if not results:
        print("⚠ 教材が1本もできませんでした")
        sys.exit(1)

# このファイルを直接実行したときだけ main() を動かす
if __name__ == "__main__":
    main()
