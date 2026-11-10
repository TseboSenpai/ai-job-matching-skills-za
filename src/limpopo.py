"""Shared helpers for the Limpopo job-matching notebooks (03 onwards)."""
import re
import pandas as pd

# --------------------------------------------------------------------------- files
def read_csv_any_encoding(path, **kw):
    """Read a CSV saved as UTF-8 or by Excel/Numbers on a Mac (Mac Roman)."""
    for enc in ("utf-8-sig", "mac_roman"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc, **kw).fillna("")
        except UnicodeDecodeError:
            continue

# --------------------------------------------------------------------------- local vocabulary
def alias_to_regex(alias):
    alias = alias.strip()
    if alias.lower().startswith("re:"):
        return alias[3:]
    body = r"\s+".join(re.escape(w) for w in alias.split())
    return rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])"

class LocalVocab:
    """Extractor for South African credentials and terms (see notebook 02)."""
    TERM_CATS = ("system", "legislation", "framework")

    def __init__(self, vocab_df, equiv_df):
        self.vocab = vocab_df
        self.label = dict(zip(vocab_df["id"], vocab_df["label"]))
        self.category = dict(zip(vocab_df["id"], vocab_df["category"]))
        self.aliases = {r["id"]: [a.strip() for a in r["aliases"].split(";")
                                  if a.strip() and not a.strip().lower().startswith("re:")]
                        for _, r in vocab_df.iterrows()}
        self.matchers = []
        for _, r in vocab_df.iterrows():
            flags = 0 if r["case_sensitive"] == "1" else re.IGNORECASE
            pats = [alias_to_regex(a) for a in r["aliases"].split(";") if a.strip()]
            self.matchers.append((r["id"], re.compile("|".join(f"(?:{p})" for p in pats), flags)))
        self.rules_all, self.rules_verified = {}, {}
        for _, r in equiv_df.iterrows():
            self.rules_all.setdefault(r["satisfied_by_id"], set()).add(r["requirement_id"])
            if r["verified"] == "1":
                self.rules_verified.setdefault(r["satisfied_by_id"], set()).add(r["requirement_id"])

    def find(self, text):
        return {vid for vid, rx in self.matchers if rx.search(text)}

    def closure(self, creds, verified_only=True):
        rules = self.rules_verified if verified_only else self.rules_all
        met, stack = set(creds), list(creds)
        while stack:
            for req in rules.get(stack.pop(), ()):
                if req not in met:
                    met.add(req); stack.append(req)
        return met

    def satisfies(self, creds, required, verified_only=True):
        return required in self.closure(creds, verified_only)

# --------------------------------------------------------------------------- requirements
EDU_CATS = {"school", "tvet", "higher", "trade"}
OPTIONAL_WORDS = re.compile(r"advantage|advantageous|preferred|desirable|recommended|an asset|would be beneficial", re.I)
NOT_ELIGIBILITY = {"MIN_FITNESS", "SETA_CERT"}            # assumed / not a personal credential
SPECIFIC_LICENCES = {"LIC_B", "LIC_EB", "LIC_C1", "LIC_C", "LIC_EC"}

def requirements_text(description):
    m = re.search(r"Requirements:(.*?)(?:Duties:|$)", description, flags=re.S)
    return m.group(1) if m else ""

def parse_requirements(description, lv):
    """Return mandatory credential requirements as a list of OR-groups, e.g. [['GR12'], ['DIPLOMA','DEGREE'], ['LIC_ANY']]."""
    groups = []
    for clause in re.split(r";", requirements_text(description)):
        if not clause.strip() or OPTIONAL_WORDS.search(clause):
            continue
        ids = {i for i in lv.find(clause)
               if lv.category.get(i) not in LocalVocab.TERM_CATS and i not in NOT_ELIGIBILITY}
        if ids & SPECIFIC_LICENCES:
            ids.discard("LIC_ANY")
        has_or = bool(re.search(r"\bor\b", clause))
        edu = sorted(i for i in ids if lv.category[i] in EDU_CATS)
        lic = sorted(i for i in ids if lv.category[i] == "licence" and i != "PRDP")
        other = sorted(ids - set(edu) - set(lic))
        for block in (edu, lic):
            if not block:
                continue
            if has_or and len(block) > 1:
                groups.append(block)
            else:
                groups.extend([[i] for i in block])
        groups.extend([[i] for i in other])
    # de-duplicate groups, keep order
    seen, out = set(), []
    for g in groups:
        k = tuple(g)
        if k not in seen:
            seen.add(k); out.append(g)
    return out

def is_eligible(creds, groups, lv, verified_only=True):
    met = lv.closure(creds, verified_only)
    return all(any(g in met for g in grp) for grp in groups)

# --------------------------------------------------------------------------- geography
# District adjacency in Limpopo (shared borders). Researcher assumption: check against a map.
ADJACENT = {
    "Capricorn":  {"Vhembe", "Mopani", "Sekhukhune", "Waterberg"},
    "Vhembe":     {"Capricorn", "Mopani"},
    "Mopani":     {"Vhembe", "Capricorn", "Sekhukhune"},
    "Sekhukhune": {"Capricorn", "Mopani", "Waterberg"},
    "Waterberg":  {"Capricorn", "Sekhukhune"},
}
DISTRICTS = list(ADJACENT)
HOME_TOWNS = {
    "Capricorn":  ["Polokwane", "Seshego", "Mankweng", "Lebowakgomo", "Senwabarwana", "Botlokwa", "Mogwadi"],
    "Vhembe":     ["Thohoyandou", "Makhado", "Musina", "Malamulele", "Sibasa", "Elim"],
    "Mopani":     ["Tzaneen", "Giyani", "Phalaborwa", "Nkowankowa", "Modjadjiskloof", "Hoedspruit"],
    "Sekhukhune": ["Jane Furse", "Groblersdal", "Burgersfort", "Marble Hall", "Steelpoort"],
    "Waterberg":  ["Mokopane", "Lephalale", "Modimolle", "Bela-Bela", "Thabazimbi"],
}

def reachable_districts(home, mobility):
    if mobility == "local":
        return {home}
    if mobility == "regional":
        return {home} | ADJACENT[home]
    return set(DISTRICTS) | {"Unknown"}            # 'province': willing to relocate anywhere in Limpopo

# --------------------------------------------------------------------------- job families
def assign_family(title, rules):
    """rules: DataFrame with columns family, title_pattern (regex), in priority order."""
    t = title.lower()
    for _, r in rules.iterrows():
        if re.search(r["title_pattern"], t):
            return r["family"]
    return "UNASSIGNED"

# --------------------------------------------------------------------------- ranking metrics
import numpy as np

def dcg(rels):
    rels = np.asarray(rels, dtype=float)
    return float(((2 ** rels - 1) / np.log2(np.arange(2, len(rels) + 2))).sum())

def ndcg_at_k(ranked_rels, all_rels, k=10):
    """ranked_rels: relevance of the top-ranked jobs in order; all_rels: relevance of every job for this candidate."""
    ideal = dcg(sorted(all_rels, reverse=True)[:k])
    return dcg(ranked_rels[:k]) / ideal if ideal > 0 else np.nan

def precision_at_k(ranked_rels, k=10, threshold=2):
    return float(np.mean(np.asarray(ranked_rels[:k]) >= threshold))

def recall_at_k(ranked_rels, all_rels, k=10, threshold=2):
    total = int(np.sum(np.asarray(all_rels) >= threshold))
    return float(np.sum(np.asarray(ranked_rels[:k]) >= threshold) / total) if total else np.nan
