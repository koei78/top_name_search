from selenium import webdriver
from bs4 import BeautifulSoup
import requests
import json
import requests, re
from lxml import html
from flask import Flask, request, jsonify
import os


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
class OpenRouterClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        print(self.api_key)
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer " + self.api_key,
            "Content-Type": "application/json"
        }

    def chat(self, system_prompt, user_payload):
        """
        system_prompt: str（LLM指示）
        user_payload: dict または str（LLMに渡すデータ）
        """

        # dict なら JSON化する
        if isinstance(user_payload, dict):
            user_content = json.dumps(user_payload, ensure_ascii=False)
        else:
            user_content = user_payload

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        }

        res = requests.post(self.url, headers=self.headers, json=data)

        if res.status_code != 200:
            raise Exception(f"APIエラー: {res.status_code} {res.text}")

        return res.json()["choices"][0]["message"]["content"]


class ResearchAI:
    def __init__(self, shopname, shopaddress,key):
        # ========== 検索・LLM 初期化 ==========

        self.shopname = shopname
        self.shopaddress = shopaddress

        self.api_key = key
        self.model = "openai/gpt-oss-20b:free"

        self.client = OpenRouterClient(self.api_key, self.model)

    # ========== 共通：ページ取得 ==========

    def page_get(self, urls):
        pages = []
        for url in urls:
            print("fetch:", url)
            try:
                res = requests.get(url, headers=HEADERS, timeout=10)
                res.raise_for_status()

                if res.encoding is None:
                    res.encoding = res.apparent_encoding

                html_text = res.text
                soup = BeautifulSoup(html_text, "html.parser")
                clean_text = soup.get_text(separator="\n")
                clean_text = clean_text[:15000]

                pages.append({
                    "url": url,
                    "text": clean_text,
                })

            except Exception as e:
                print(f"⚠ {url} の取得に失敗: {e}")

        return pages

    # ========== 直接「店舗代表者」を抜く系 ==========

    def parse_direct_rep_from_json(self, json_text):
        """
        serch_name 用：LLMのJSON返答を解析して、
        ・confidence >= 0.80 かつ has_representative_info == True の代表者を返す
        ・なければ False を返す
        """
        data = json.loads(json_text)
        pages = data.get("pages", [])

        for p in pages:
            conf = p.get("confidence", 0.0)
            if conf >= 0.80 and p.get("has_representative_info") and p.get("representative_name"):
                return {
                    "name": p.get("representative_name"),
                    "title": p.get("representative_title"),
                    "company": p.get("company_name"),
                    "url": p.get("url"),
                }

        return False

    def serch_name(self):
        """
        店名＋住所から、その店舗の「代表者／オーナー／店主」を直接探すフェーズ
        """
        shopname = self.shopname
        shopaddress = self.shopaddress


        resp = requests.get(
            "https://ecosia1-477268798017.europe-west1.run.app/search",
            params={"q": f"{shopname} {shopaddress} 代表 オーナー 店主", "top_n": 3}
        )

        links = resp.json().get("links", [])

        print(links)        # リストそのまま

        pages = self.page_get(links)

        prompt = """あなたは日本の店舗情報を解析するアシスタントです。

与えられた複数のWebページのテキストから、
「指定された店舗」の情報かどうかを判定し、
もし代表者名・オーナー名・店主など、その店舗のトップに関する情報があれば抽出して報告してください。

【やること】

1. 各ページごとに、そこに書かれている店舗が
   target_shop の「店名」と「住所」と同一の店かどうかを判定してください。
   - 完全一致でなくても構いませんが、
     店名と住所の両方について、文脈的にほぼ同一店舗と判断できる場合のみ true としてください。
   - チェーン店や類似名の別店舗の場合は false にしてください。
   - 「テナント募集」「前テナント」「過去に入居していた店舗」などは false にしてください。

2. 対象店舗と一致すると判断したページについてのみ、
   以下のような「店のトップ」に関する情報を探してください。
   - 代表者
   - 代表者名
   - 代表
   - 代表取締役
   - オーナー
   - 店主
   - マスター
   - 経営者
   など、それに相当する表現。

   ただし、以下は対象外です：
   - グルメサイト（食べログ等）の運営会社の代表者
   - HP制作会社・システム会社の代表者
   - 不動産会社・管理会社の担当者・代表者
   - 取材記事の「記者」「ライター」「編集者」
   - 個人紹介だが店との関係が明確でない人

3. 見つかった場合は、
   - 個人名（代表者名・オーナー名など）
   - 会社名（株式会社○○ など運営法人。分かる範囲で）
   - その情報が載っていた原文の抜粋（周辺数行）
   を抽出してください。

4. 情報があいまい、推測レベル、別店舗の可能性が高い場合は、
   is_match を false にし、代表者情報は抽出しないでください。

5. 出力は、必ず以下の JSON 形式で返してください。
   それ以外の文章は一切書かないでください。

【入力形式（論理的構造）】

- target_shop:
  - name: 店名（文字列）
  - address: 住所（文字列）

- pages: 最大3件までのページ情報リスト。各要素は以下の形式です。
  - url: ページURL
  - text: ページ本文のテキスト（HTMLから抽出済み）

【出力形式（必ずこのJSONのみ）】

{
  "target_shop": {
    "name": "string",
    "address": "string"
  },
  "pages": [
    {
      "url": "string",
      "is_match": true or false,
      "reason": "string",
      "has_representative_info": true or false,
      "representative_name": "string or null",
      "representative_title": "string or null",
      "company_name": "string or null",
      "raw_snippet": "string or null",
      "confidence": 0.0
    }
  ],
  "has_any_representative_info": true or false
}
"""

        messages = {
            "target_shop": {
                "name": shopname,
                "address": shopaddress
            },
            "pages": pages
        }

        response_text = self.client.chat(prompt, messages)
        rep = self.parse_direct_rep_from_json(response_text)
        return rep

    # ========== 法人系：共通ヘルパー ==========

    def get_pages_text(self, links):
        pages = self.page_get(links)
        pages_text = "\n\n".join(
            [f"[{i+1}] URL: {p['url']}\n{p['text']}" for i, p in enumerate(pages)]
        )
        return pages_text, pages

    def is_corporate_name(self, name):
        """
        文字列が「法人名っぽいか」をざっくり判定する。
        """
        if not name:
            return False

        corp_keywords = [
            "株式会社",
            "合同会社",
            "有限会社",
            "医療法人",
            "社会福祉法人",
            "学校法人",
            "NPO法人",
            "特定非営利活動法人",
        ]

        for k in corp_keywords:
            if k in name:
                return True

        return False

    # ========== 店舗 → 運営法人名 抽出 ==========

    def extract_company_name(self):
        shopname = self.shopname
        shopaddress = self.shopaddress

        # ------------ 検索リンク取得（運営会社用）------------
        resp = requests.get(
            "https://ecosia1-477268798017.europe-west1.run.app/search",
            params={"q": f"{shopname} {shopaddress} 運営会社", "top_n": 3}
        )

        links = resp.json().get("links", [])
        pages_text, _ = self.get_pages_text(links)

        # ------------ 法人名抽出用プロンプト ------------
        step1_company_prompt = f"""
あなたは日本の店舗情報を「非常に厳格な基準」で精密に解析するAIエージェントです。

以下には、複数のWebページから抽出されたテキストが含まれています。
これらは必ずしも同じ店舗の情報とは限りません。
また、ページ内には無関係な法人名・サイト運営会社名・他店舗の情報が混在している可能性があります。

あなたの最重要ミッションは、
「対象店舗 *だけ* の運営法人名を、誤検出なしで特定すること」です。
あいまいな場合は、無理に法人名を決めず、ルールに従って安全側に倒れてください。

---

【対象店舗】
- 店名: {shopname}
- 住所: {shopaddress}

---

【全体ルール（絶対遵守）】

- 対象店舗と無関係なページ内容・法人名はすべて無視すること
- 想像で法人名を作らないこと（補完・創作は禁止）
- ページに書かれていない法人名を推測で書かないこと
- 「食べログ」「ぐるなび」「ホットペッパー」などのグルメサイト運営会社名を
  対象店舗の法人名として絶対に採用しないこと
- クレジットカード会社・決済代行業者・ビルオーナー・広告代理店などの
  第三者企業名も、対象店舗の法人名として採用してはいけない

---
【このようなページも「対象店舗に関連するページ」として扱う】

以下のような文章が含まれている場合は、
住所が {shopaddress} と完全一致していなくても、
対象店舗に関連するページとして扱ってよい：

例：
「麺JAPAN株式会社（本社：東京都新宿区…）は、
 東京都東村山市の店舗『新潟発祥 なおじ東村山店』及び
 『東村山 ごちそうや ぽっ蔵』において、新商品◯◯の販売を開始します。」

このように、

- 冒頭に法人名（株式会社◯◯ など）があり、
- 文中に対象店舗の店名（{shopname}）が明示されていて、
- 「◯◯株式会社は、◯◯店において…」という構造になっている場合、

その法人名は対象店舗の「運営会社の有力候補」として扱ってよい。

=====================
【ステップ1：一致判定（最重要）】
=====================

各ページ（テキスト）ごとに、
「そのページが対象店舗 *だけ* に関する情報か」を厳密に判定してください。

以下の情報から一致度を総合的に判断します：

- 店名の一致
  - 完全一致
  - 表記ゆれ（ひらがな/カタカナ/漢字/英字・全角半角）も考慮
  - 「○○店」「○○本店」などの枝番表現も考慮

- 住所の一致
  - 都道府県・市区町村レベルで対象住所と一致していること
  - 丁目・番地・号が概ね一致していること
  - ビル名・フロアの違いは許容（同一ビル内の別テナントは要注意）

- 電話番号
  - 対象店舗の電話番号と完全一致していれば、非常に強い一致根拠
  - 番号が異なる場合、そのページは別店舗の可能性が高い

- 店舗特徴・文脈
  - メニュー内容
  - 価格帯
  - 営業時間
  - 席数・内装の特徴
  - 口コミ内容などが、対象店舗と矛盾していないか

【チェーン店・同名店舗への注意】
- 同じ店名で複数の住所が出てくる場合、
  対象住所 {shopaddress} と異なる住所のページは「別店舗」とみなし、一致しないと判断すること。
- 「◯◯（新宿店）」「◯◯（渋谷店）」のように支店名がある場合も、
  対象住所と一致しない店舗は一致対象から外すこと。

【ステップ1の結論】
- 対象店舗と明確に紐づくと判断できたページだけを「一致したページ」とする
- 一致しているか判断できない曖昧なページは「一致していない」とみなし、完全に無視する

一致していないページは、
そのページ内にどんな法人名が書かれていても、絶対に使ってはいけません。

---

=====================
【ステップ2：法人名の抽出（対象店舗に一致したページのみ）】
=====================

ステップ1で「一致した」と判断できたページの中だけを使い、
その店舗の「運営会社（法人）」に関する情報を探してください。

探すべき記述の例：

- 「運営会社：◯◯」
- 「会社概要」「会社情報」「事業者」「法人名」「運営事業者」
- 「株式会社◯◯」「合同会社◯◯」「有限会社◯◯」
- 「◯◯株式会社」のように社名が前後反転した表記
- 「◯◯を運営する株式会社△△」のように、店舗名と法人名が紐づいている記述

【強く採用すべきパターン】
- 法人名の近くに、対象店舗の店名（{shopname}）や
  「当店」「本店舗」「◯◯店」といった表現がある
- 「店舗情報」「会社概要」など、明らかにその店の運営会社を説明している箇所に法人名が書かれている

【絶対に採用してはいけない法人名の例】
- グルメサイト・予約サイト・口コミサイト・ポータルサイトの運営会社
  - 例：「このサイトを運営する株式会社◯◯」「食べログを運営する株式会社カカクコム」など
- 決済サービス・クレジットカード会社・ポイントサービス会社
- 配送業者（○○運輸など）
- 広告代理店・制作会社（サイトを制作した会社など）
- まったく別の店舗（支店・系列店を含む）の法人名
- 単なる挨拶や取引先紹介に出てくる他社名

【複数候補がある場合の扱い】
- 対象店舗ともっとも強く結びついた法人名を 1 つだけ選んでください。
  - 店舗名との近接
  - 「運営会社」「事業者」「会社概要」などの語との近接
  - 対象住所との一致 などを総合して判断
- どの法人名が対象店舗のものか明確に判断できない場合、
  「法人名は特定不能」とみなし、ステップ3のルール2または3に従ってください。
  （あいまいな候補を無理に result に入れてはいけません）

---

=====================
【ステップ3：最終判断】
=====================

以下のルールに従って、最終的な "result" を 1 つだけ決定してください。

1. 対象店舗に一致したページの中に、
   「対象店舗の運営会社」であると明確に判断できる法人名がある場合
   → その法人名をそのまま result に入れる

2. 対象店舗に一致したページはあるが、
   - 法人名の記載が見つからない
   - または、どの法人名が対象店舗のものか判別できない（候補が曖昧）
   → 店舗名（{shopname}）を result に入れる

3. すべてのページが対象店舗に一致しない場合
   → "False" を result に入れる

---

【出力上の厳守事項】

- 出力は以下の JSON 形式のみとすること。
- 説明文・コメント・推論過程など、JSON以外の文字列は一切出力してはいけません。
- result には次のいずれかのみを入れてください：
  - 抽出した法人名（完全な法人名）
  - 店舗名（{shopname}）
  - "False"

【出力形式（必ずこのJSONだけを返す）】

{{
  "result": "string"   // 法人名 or 店舗名 or "False"
}}

---

【ページ内容】
{pages_text}
"""

        response_text = self.client.chat(step1_company_prompt, pages_text)

        try:
            data = json.loads(response_text)
            company_result = data.get("result", "").strip()
        except Exception:
            company_result = ""

        return company_result

    # ========== 法人 → 代表者 抽出 ==========

    def extract_corp_representative(self, company_name):
        """
        法人名がわかったあとに、その法人の代表者名をリサーチするフェーズ
        """
        if not company_name or company_name == "False":
            return ""

        # ------------ 検索リンク取得（代表者用）------------
        resp = requests.get(
            "https://ecosia1-477268798017.europe-west1.run.app/search",
            params={"q": f"{company_name} 代表取締役 OR 代表者 OR 代表社員 OR 代表理事 会社概要", "top_n": 5}
        )

        links = resp.json().get("links", [])
        pages_text, _ = self.get_pages_text(links)

        # ------------ 代表者抽出用プロンプト ------------
        step2_rep_prompt = f"""
あなたは日本の法人情報を精密に解析するAIエージェントです。

以下には、法人「{company_name}」に関する複数のWebページから抽出されたテキストが含まれています。
これらには、会社概要・代表者挨拶・採用情報・ニュース記事・取引先の紹介など、
さまざまな情報が混在している可能性があります。

あなたの最重要ミッションは、
「法人 {company_name} の現在の代表者（代表取締役・代表社員・代表理事など）の氏名」を
できる限り正確に1名だけ特定することです。

---

【全体ルール（絶対遵守）】

- 想像で名前を作らないこと（補完・創作は禁止）
- 法人 {company_name} と無関係な人物名はすべて無視すること
- 過去の役職者・創業者・相談役・顧問が出てきても、
  現在の代表者と明確に書かれていない場合は採用しないこと
- 他社の代表者名・取引先の担当者名・インタビュー対象者の名前は採用してはいけない

---

【重要】
明らかに企業名が{company_name}出ない場合は、そのページは無視することその代表者名も違う
【代表者として採用してよい記述の例】

- 「代表取締役社長　山田太郎」
- 「代表取締役　山田太郎」
- 「代表者名：山田太郎」
- 「代表社員　山田太郎」
- 「代表理事　山田太郎」
- 「法人 {company_name}　代表　山田太郎」

【出力仕様】

- 代表者が特定できる場合
  → 代表者のフルネームだけを result に入れる（例："山田太郎"）
- 特定できない／情報がない場合
  → "Unknown" を result に入れる

【出力形式（必ずこのJSONだけを返す）】

{{
  "result": "string"   // 代表者名 or "Unknown"
}}

---

【ページ内容】
{pages_text}
"""

        rep_response_text = self.client.chat(step2_rep_prompt, pages_text)

        try:
            data = json.loads(rep_response_text)
            rep_name = data.get("result", "").strip()
        except Exception:
            rep_name = ""

        return rep_name

    # ========== 追加：インボイス登録番号 抽出 ==========

    def extract_invoice_number(self, company_name):
        """
        指定した店舗（店名＋住所）と、わかっていれば法人名を手がかりにして
        適格請求書発行事業者の登録番号（インボイス番号）を探す。

        戻り値:
          - 見つかった場合: "T1234567890123" などの文字列
          - 見つからない/曖昧: ""（空文字）
        """
        shopname = self.shopname
        shopaddress = self.shopaddress

        # -------- 検索クエリを組み立て --------
        query_parts = []
        if company_name and company_name != "False" and company_name != shopname:
            query_parts.append(company_name)

        query_parts.extend([
            shopname,
            shopaddress,
            "適格請求書発行事業者",
            "インボイス",
            "登録番号"
        ])

        query = " ".join(query_parts)
        print("=== インボイス検索クエリ ===", query)

        resp = requests.get(
            "https://ecosia1-477268798017.europe-west1.run.app/search",
            params={"q": query, "top_n": 3}
        )

        links = resp.json().get("links", [])
        pages_text, _ = self.get_pages_text(links)

        # -------- LLM プロンプト --------
        invoice_prompt = f"""
あなたは日本の税務情報・インボイス制度に詳しいAIエージェントです。

以下には、店舗およびその運営法人に関する複数のWebページから抽出されたテキストが含まれています。
あなたのミッションは、指定された店舗/法人に対応する
「適格請求書発行事業者の登録番号（インボイス番号）」を、誤検出なく特定することです。

【対象店舗】
- 店名: {shopname}
- 住所: {shopaddress}

【運営法人候補】
- 法人名候補: {company_name if company_name else "（不明）"}

【探すべき情報】

- 適格請求書発行事業者の登録番号
- 一般的には「T」+ 13桁の数字の形式（例：T1234567890123）
- 「登録番号」「インボイス」「適格請求書発行事業者」などの語の近くに書かれていることが多い

【絶対にやってはいけないこと】

- 店名や法人名だけから、番号を推測・創作してはいけない
- 他社のインボイス番号を、この店舗/法人の番号として流用してはいけない
- 決済代行会社・グルメサイト運営会社・不動産会社など、
  関係のない第三者企業のインボイス番号を採用してはいけない

【採用してよい例】

- 「適格請求書発行事業者登録番号：T1234567890123」
- 「登録番号 T1234567890123」
- 「当社（株式会社◯◯）のインボイス登録番号は T1234567890123 です。」

このとき、文脈上 「株式会社◯◯」 が {company_name} である、
または対象店舗（{shopname}）を運営している会社であると判断できる場合にのみ、
その番号を result に採用してください。

【あいまいな場合の扱い】

- 複数のインボイス番号候補があり、どれが対象法人かわからない場合
- 店名・住所・法人名との対応関係がはっきりしない場合

→ 無理に番号を選ばず、"Unknown" を result にしてください。

【出力形式（必ずこのJSONだけを返す）】

{{
  "result": "string"   // インボイス登録番号（例: "T1234567890123"）または "Unknown"
}}

---

【ページ内容】
{pages_text}
"""

        response_text = self.client.chat(invoice_prompt, "")
        try:
            data = json.loads(response_text)
            invoice = data.get("result", "").strip()
        except Exception:
            invoice = ""

        # "Unknown" や空文字は「見つからず」と扱う
        if not invoice or invoice.lower() == "unknown":
            return ""

        return invoice

    # ========== インボイス番号 → 法人名（＋α） ==========

    def get_corp_info_from_invoice(self, invoice_number):
        """
        インボイス番号から法人番号サイトのページを開いて、
        ・法人名（company_name）
        ・法人番号（corporate_number）
        ・（取れれば）代表者名 representative
        ・参照元URL（source_url）
        を dict で返す。

        何も取れなかった場合は None を返す。
        """
        if not invoice_number:
            return None

        # インボイス番号から数字だけ抜き出し（T1234... → 1234...）
        corporate_number = ''.join(re.findall(r'\d', invoice_number))
        if not corporate_number:
            return None

        source_url = f"https://www.houjin-bangou.nta.go.jp/henkorireki-johoto.html?selHouzinNo={corporate_number}"

        try:
            r = requests.get(source_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            r.encoding = "utf-8"
            t = html.fromstring(r.text)

            # いままで使っていた XPath をそのまま利用
            texts = t.xpath("/html/body/div[1]/form/div[3]/main/div/div[1]/dl/dd[2]/text()")
            company_name = texts[0].strip() if texts else None

            if not company_name:
                # ページ取得できたけど法人名が抜けなかった場合
                return {
                    "company_name": None,
                    "corporate_number": corporate_number,
                    "representative": None,
                    "source_url": source_url,
                }

            # 代表者名も取りたければここで別の dd[...] を抜く
            # 今はまだ構造不明なので None にしておく
            representative = None

            print("[get_corp_info_from_invoice] 取得した法人名:", company_name)

            return {
                "company_name": company_name,
                "corporate_number": corporate_number,
                "representative": representative,
                "source_url": source_url,
            }

        except Exception as e:
            print("⚠ 法人番号サイトの取得に失敗:", e)
            return None

    # ========== オーケストレーター：優先順位付き ==========

    def run(self):
        """
        優先順位付きリサーチAI:

          1. まず店舗そのものの代表者（オーナー・店主）を Web から直接探す
          2. 見つからなかったら 店名＋住所 からインボイス番号を探す
             2-1. インボイス番号が見つかったら
                  → 法人番号サイトから法人名・代表者を取る
                  → 法人名だけなら LLM で代表者を補完
          3. インボイス番号も見つからなかったら
             → 店舗名から運営法人名を LLM で特定
             → 法人名が法人っぽければ、LLMで代表者名を探す
        """

        shopname = self.shopname
        shopaddress = self.shopaddress

        invoice_number = ""
        corp_info = None

        # ======================
        # STEP1: 店舗の代表者を直接探索
        # ======================
        print("=== STEP1: 店舗の代表者を直接探索 ===")
        direct_rep = self.serch_name()

        if direct_rep:
            print("[STEP1] 店舗代表者を検出:", direct_rep)

            # ここでインボイスもついでに探しておく（任意）
            invoice_number = self.extract_invoice_number(
                direct_rep.get("company")
            )

            return {
                "shopname": shopname,
                "shopaddress": shopaddress,
                "company_name": direct_rep.get("company"),
                "representative": direct_rep.get("name"),
                "representative_title": direct_rep.get("title"),
                "source_url": direct_rep.get("url"),
                "invoice_number": invoice_number,
                "route": "shop_direct",
            }

        print("[STEP1] 店舗からは代表者が特定できなかった → インボイスルートへ")

        # ======================
        # STEP2: 店名＋住所 からインボイス番号を探索
        # ======================
        print("=== STEP2: 店名＋住所からインボイス番号を探索 ===")
        invoice_number = self.extract_invoice_number(
            None   # まだ法人名は使わない
        )

        if invoice_number:
            print("[STEP2] インボイス番号を検出:", invoice_number)

            # インボイス番号 → 法人番号サイトから法人情報を取得
            corp_info = self.get_corp_info_from_invoice(invoice_number)
            print("[STEP2] インボイスから得られた法人情報:", corp_info)

            if corp_info and corp_info.get("company_name"):
                company_from_invoice = corp_info["company_name"]

                # もし法人番号サイトで代表者名まで取れていれば、それをそのまま採用
                if corp_info.get("representative"):
                    return {
                        "shopname": shopname,
                        "shopaddress": shopaddress,
                        "company_name": company_from_invoice,
                        "representative": corp_info.get("representative"),
                        "representative_title": None,
                        "source_url": corp_info.get("source_url"),
                        "invoice_number": invoice_number,
                        "route": "invoice_official",
                    }

                # 法人名だけわかった場合 → LLMで代表者名を探す
                print("=== STEP2-追加: 法人名から代表者を LLM で探索 ===")
                corp_rep = self.extract_corp_representative(company_from_invoice)
                print("[STEP2-追加] LLMで推定した法人代表者:", corp_rep)

                if corp_rep and corp_rep != "Unknown":
                    return {
                        "shopname": shopname,
                        "shopaddress": shopaddress,
                        "company_name": company_from_invoice,
                        "representative": corp_rep,
                        "representative_title": None,
                        "source_url": corp_info.get("source_url"),
                        "invoice_number": invoice_number,
                        "route": "invoice_corp_representative",
                    }

                # 代表者までは取れなかったが、法人名だけは分かったケース
                return {
                    "shopname": shopname,
                    "shopaddress": shopaddress,
                    "company_name": company_from_invoice,
                    "representative": None,
                    "representative_title": None,
                    "source_url": corp_info.get("source_url"),
                    "invoice_number": invoice_number,
                    "route": "invoice_corp_only",
                }

            # インボイス番号はあるけど、法人番号サイトから法人名を取れなかった場合
            print("[STEP2] インボイス番号はあるが法人名が取れず → 法人名LLM検索へフォールバック")

        else:
            print("[STEP2] インボイス番号が見つからなかった → 法人名LLM検索へ")

        # ======================
        # STEP3: 店舗 → 運営法人名を LLM で特定
        # ======================
        print("=== STEP3: 店舗から運営法人名を LLM で探索 ===")
        company = self.extract_company_name()
        print("[STEP3] LLMから推定された company_name:", company)

        if (not company) or (company == "False"):
            print("[STEP3] 法人名が特定できなかった")

            # インボイス番号だけは残しておく（あれば）
            return {
                "shopname": shopname,
                "shopaddress": shopaddress,
                "company_name": None,
                "representative": None,
                "代表肩書き": None,
                "representative_title": None,
                "source_url": None,
                "invoice_number": invoice_number,
                "route": "no_info",
            }

        if company == shopname:
            print("[STEP3] company_name が店舗名と同じ → 法人特定できていない扱い")

            return {
                "shopname": shopname,
                "shopaddress": shopaddress,
                "company_name": company,
                "representative": None,
                "representative_title": None,
                "source_url": None,
                "invoice_number": invoice_number,
                "route": "shopname_only",
            }

        # ======================
        # STEP4: company_name が法人名っぽいかチェック
        # ======================
        print("=== STEP4: company_name が法人かどうか判定 ===")
        if not self.is_corporate_name(company):
            print("[STEP4] company_name が法人名っぽくない → 個人屋号かも。ここで終了。")

            return {
                "shopname": shopname,
                "shopaddress": shopaddress,
                "company_name": company,
                "representative": None,
                "representative_title": None,
                "source_url": None,
                "invoice_number": invoice_number,
                "route": "non_corporate_company_name",
            }

        # ======================
        # STEP5: 法人の代表者名を LLM で探索
        # ======================
        print("=== STEP5: 法人の代表者を LLM で探索 ===")
        corp_rep = self.extract_corp_representative(company)
        print("[STEP5] 法人代表者(LM推定):", corp_rep)

        if not corp_rep or corp_rep == "Unknown":
            return {
                "shopname": shopname,
                "shopaddress": shopaddress,
                "company_name": company,
                "representative": None,
                "representative_title": None,
                "source_url": None,
                "invoice_number": invoice_number,
                "route": "corp_without_rep",
            }

        return {
            "shopname": shopname,
            "shopaddress": shopaddress,
            "company_name": company,
            "representative": corp_rep,
            "representative_title": None,
            "source_url": None,
            "invoice_number": invoice_number,
            "route": "corp_representative",
        }


# ========== 動作テスト ==========

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "ResearchAI API Running"})


@app.route("/api/run", methods=["POST"])
def run_api():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSONが必要です"}), 400

    shopname = data.get("shopname")
    shopaddress = data.get("shopaddress")
    key= data.get("key")

    if not shopname or not shopaddress:
        return jsonify({"error": "shopname と shopaddress は必須です"}), 400

    # ==== ResearchAI 実行 ====
    ai = ResearchAI(shopname, shopaddress,key)
    result = ai.run()

    # ==== ログとして出力（Cloud Run のログに残る） ====
    print("\n=== リサーチ結果 ===")
    print("店舗名      :", result.get("shopname"))
    print("住所        :", result.get("shopaddress"))
    print("運営法人名  :", result.get("company_name"))
    print("代表者名    :", result.get("representative"))
    print("代表肩書き  :", result.get("representative_title"))
    print("インボイス番号:", result.get("invoice_number"))
    print("取得ルート  :", result.get("route"))
    print("ソースURL   :", result.get("source_url"))

    return jsonify(result)
@app.route("/api/add", methods=["POST"])
def run_add():
    shopname = request.form.get("shopname")
    shopaddress = request.form.get("shopaddress")
    key = request.form.get("key")
    row = request.form.get("row")
    sheet_name = request.form.get("sheet")
    file = request.files.get("file")

    if not shopname or not shopaddress or not key:
        return jsonify({"error": "shopname / shopaddress / key は必須です"}), 400
    if not row:
        return jsonify({"error": "row が指定されていません"}), 400
    if not sheet_name:
        return jsonify({"error": "sheet が指定されていません"}), 400
    if not file:
        return jsonify({"error": "file が添付されていません"}), 400

    # === 一時ファイルへ保存 ===
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        file.save(tmp.name)
        SERVICE_ACCOUNT_FILE = tmp.name

    import gspread
    from google.oauth2 import service_account

    SPREADSHEET_ID = "1CI69F1PDS2ROYLP4Q4dO37ba1MBvW72yqxx9jPy9UL4"
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=scopes
    )
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(SPREADSHEET_ID).worksheet(sheet_name)

    # ================================
    # ① AI検索前に赤色へ変更（開始マーク）
    # ================================
    ws.format(f"A{row}:H{row}", {
        "backgroundColor": {"red": 1, "green": 0.8, "blue": 0.8}
    })

    # ================================
    # ② ResearchAI 実行（ここ重い）
    # ================================
    ai = ResearchAI(shopname, shopaddress, key)
    result = ai.run()

    # === 出力デバッグ ===
    print("\n=== リサーチ結果 ===")
    for k, v in result.items():
        print(k, ":", v)

    # ================================
    # ③ safe() で値整形
    # ================================
    def safe(v):
        if v is None:
            return "不明"
        if isinstance(v, str):
            s = v.strip()
            if s == "" or s.lower() in ["unknown", "false", "none", "null"]:
                return "不明"
            return s
        return str(v)

    values = [
        safe(result.get("company_name")),
        safe(result.get("representative")),
        safe(result.get("representative_title")),
        safe(result.get("invoice_number")),
        safe(result.get("route")),
        safe(result.get("source_url")),
    ]

    # ================================
    # ④ 値書き込み
    # ================================
    ws.update(f"C{row}:H{row}", [values])

    # ================================
    # ⑤ 白へ戻す（完了マーク）
    # ================================
    ws.format(f"A{row}:H{row}", {
        "backgroundColor": {"red": 1, "green": 1, "blue": 1}
    })

    result["sheet_write"] = {
        "status": "success",
        "row": row,
        "range": f"C{row}:H{row}"
    }

    return jsonify(result)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

