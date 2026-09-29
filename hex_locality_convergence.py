"""
hex_locality_convergence.py
============================
تجربة "ضابط المحلية + بروتوكول التقارب" — الأولويتان #1 و#3 من جدول الإصلاح
بتقرير النقد الذاتي (القسم 13)، مبنية فوق hex_killtest_final.py الأصلي.

الفروقات الأساسية عن الكود الأصلي (كلها لمعالجة عيوب مكتشفة سابقاً):
  1. مجموعة تحقق (validation) منفصلة + إيقاف مبكر (early stopping) — يحل
     مشكلة "EPOCHS=3 ثابتة بدون قياس تقارب" اللي قلبت النتيجة بمقدار 2.5x.
  2. منحنى تعلّم مُسجَّل لكل تشغيلة — تقدر تشوف بعينك هل النموذج تقارب فعلاً
     أو لسا يتحسّن (بدل الثقة برقم نقطة واحدة).
  3. شرط جديد local_random — رسم عشوائي منتظم الدرجة لكنه مقيّد بمحلية مكانية
     (Chebyshev distance <= LOCALITY_RADIUS على التوروس)، لفصل "المحلية" عن
     "الشكل السداسي تحديداً" (الالتباس الثلاثي، القسم 7 بالتقرير النقدي).
  4. AveragePooling حقيقي (F.avg_pool2d) بدل Resize ثنائي الخطي الافتراضي.
  5. جملة الحكم النهائي تستخدم فرق النسب المئوية مباشرة (لا عتبة كسرية
     عشوائية) — يتفادى باگ "hex_cons > hex_perm + 1.0" السابق.

⚠️ ملاحظة أمانة: هذا الكود اتُحقق منطقياً وبُني على نفس المعمارية المُثبتة
بالتقارير السابقة، لكنه **لم يُشغَّل فعلياً هون** — بيئة الحاوية ما فيها GPU
ولا مساحة قرص كافية لتثبيت PyTorch+CUDA (تحققت من هذا فعلياً، أرفقت التفاصيل
بالرد). المكوّن الوحيد اللي تحقق منه فعلياً بالتشغيل هو بناء التوبولوجيا
(topology_check.py المرفق) لأنه ما يحتاج PyTorch. لازم تشغّل هذا الملف على
جهازك (RTX 3070) وتتأكد من النتائج بنفسك — هذا نفس المبدأ اللي حددته إنت
بملف التسليم الأصلي: "أي كود جديد يجب اختباره وظيفياً فعلياً قبل التسليم".
أنا التزمت بالجزء اللي قدرت أتحقق منه فعلياً، وبقية الالتزام عليك بجهازك.
"""

import argparse
import json
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

# ---------------------------------------------------------------- إعدادات
GRID = 14
NUM_NODES = GRID * GRID
DEGREE = 6
HIDDEN_DIM = 16
NUM_LAYERS = 12
ALPHA = 0.3
K_BASE = 2.5
TOP_K = 3
LOCALITY_RADIUS = 3          # نتيجة topology_check.py: أول radius نجح ببناء درجة=6 كاملة
MAX_EPOCHS = 60
PATIENCE = 6                 # إيقاف مبكر: توقف لو val_acc ما تحسّنت 6 دورات متتالية
BATCH_SIZE = 128
LR = 1e-3
MODEL_SEEDS = [0, 1, 2, 3, 4]
TOPOLOGY_SEEDS = [1, 7, 13, 42, 99, 5, 21, 88]   # 8 بذور، قابل للرفع لـ19 لاحقاً
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------- بناء التوبولوجيا
def hex_offsets():
    return [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]


def idx(q, r):
    return (q % GRID) * GRID + (r % GRID)


def coords(i):
    return divmod(i, GRID)


def torus_dist(a, b):
    qa, ra = coords(a); qb, rb = coords(b)
    dq = min(abs(qa - qb), GRID - abs(qa - qb))
    dr = min(abs(ra - rb), GRID - abs(ra - rb))
    return max(dq, dr)


def build_hex_slots():
    slots = {}
    for q in range(GRID):
        for r in range(GRID):
            i = idx(q, r)
            slots[i] = [idx(q + dq, r + dr) for dq, dr in hex_offsets()]
    return slots


def build_hex_perm_slots(hex_slots, seed=123):
    rng = random.Random(seed)
    perm = {}
    for node, neigh in hex_slots.items():
        shuffled = neigh[:]
        rng.shuffle(shuffled)
        perm[node] = shuffled
    return perm


_LOCAL_CANDIDATES_CACHE = {}
_LOCAL_EDGES_CACHE = {}


def _local_candidate_edges(radius):
    """يُحسب مرة وحدة ويُخزّن مؤقتاً — لا علاقة له بالعشوائية، فقط هندسة الشبكة."""
    if radius not in _LOCAL_EDGES_CACHE:
        candidates = {
            i: [j for j in range(NUM_NODES) if j != i and torus_dist(i, j) <= radius]
            for i in range(NUM_NODES)
        }
        edges = [(i, j) for i in range(NUM_NODES) for j in candidates[i] if j > i]
        _LOCAL_CANDIDATES_CACHE[radius] = candidates
        _LOCAL_EDGES_CACHE[radius] = edges
    return _LOCAL_EDGES_CACHE[radius]


def build_local_random_slots(seed, radius=LOCALITY_RADIUS, max_restarts=5000):
    """
    رسم منتظم الدرجة 6، كل الأضلاع بمسافة <= radius على التوروس.

    ⚠️ تصحيح (2026-09-23): النسخة الأولى كانت تبني الرسم عقدة-عقدة (كل عقدة
    تكمل جيرانها دفعة وحدة) — فشلت فعلياً على جهاز المستخدم ببذرة seed=1 رغم
    نجاحها بالصدفة ببذرة seed=7 بالاختبار الأول. السبب: عقد تُعالَج أولاً
    "تسرق" فرص عقد مجاورة تُعالَج لاحقاً (تشبّع مبكر). الإصلاح: نبني على
    مستوى **الأضلاع** (نخلط كل الأضلاع المرشحة ونضيفها بالتتابع لو ما كسرت
    قيد الدرجة)، وهذا أوثق بكثير إحصائياً. تحقق فعلياً: نجح على كل البذور
    الثمانية المستخدمة بهذا الملف (TOPOLOGY_SEEDS) بأقل من 2000 محاولة لكل
    بذرة، بأقل من ثانية واحدة إجمالاً.
    """
    edges_pool = _local_candidate_edges(radius)
    for restart in range(max_restarts):
        rng = random.Random(seed * 1_000_003 + restart)
        edges = edges_pool[:]
        rng.shuffle(edges)
        deg = [0] * NUM_NODES
        adj = [set() for _ in range(NUM_NODES)]
        for i, j in edges:
            if deg[i] < DEGREE and deg[j] < DEGREE and j not in adj[i]:
                adj[i].add(j); adj[j].add(i)
                deg[i] += 1; deg[j] += 1
        if all(d == DEGREE for d in deg):
            return {i: sorted(adj[i]) for i in range(NUM_NODES)}
    raise RuntimeError(f"فشل بناء local_random (seed={seed}, radius={radius}) بعد {max_restarts} محاولة — جرّب radius أكبر")


def build_global_random_slots(seed):
    import networkx as nx
    G = nx.random_regular_graph(DEGREE, NUM_NODES, seed=seed)
    return {i: sorted(G.neighbors(i)) for i in range(NUM_NODES)}


def slots_to_tensor(slots):
    """(NUM_NODES, DEGREE) tensor of neighbor indices, ترتيب المواضع كما بُني."""
    arr = np.zeros((NUM_NODES, DEGREE), dtype=np.int64)
    for i in range(NUM_NODES):
        arr[i] = slots[i]
    return torch.from_numpy(arr)


# ------------------------------------------------------------------ النموذج
class GatedGeomGNN(nn.Module):
    def __init__(self, neighbor_idx: torch.Tensor, positional: bool, no_gate: bool = False):
        super().__init__()
        self.register_buffer("neighbor_idx", neighbor_idx)  # (N, 6)
        self.positional = positional
        self.no_gate = no_gate
        d = HIDDEN_DIM
        self.input_proj = nn.Linear(1, d)
        if positional:
            self.W = nn.Parameter(torch.empty(DEGREE, d, d))
            for k in range(DEGREE):
                nn.init.xavier_uniform_(self.W[k])
        else:
            self.msg_lin = nn.Linear(d, d)
        self.gate_mlp = nn.Sequential(nn.Linear(2 * d, d // 2), nn.ReLU(), nn.Linear(d // 2, 1))
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(NUM_LAYERS)])
        self.readout = nn.Linear(d, 10)

    def message(self, h):
        # h: (B, N, d) -> gathered neighbor features (B, N, 6, d)
        nbr = h[:, self.neighbor_idx]  # (B, N, 6, d)  broadcasts over batch correctly since
        # h shape is (B,N,d); indexing h[:, idx_tensor] on dim=1 works via advanced indexing
        if self.positional:
            # كل موضع k له مصفوفة مستقلة W[k]
            msg = torch.einsum('bnkd,kde->bnke', nbr, self.W)
        else:
            msg = self.msg_lin(nbr)
        return nbr, msg

    def forward(self, x):
        # x: (B, NUM_NODES) قيم بكسل بين 0 و1
        B = x.shape[0]
        h0 = self.input_proj(x.unsqueeze(-1))  # (B, N, d)
        h = h0
        for layer in range(NUM_LAYERS):
            h_center = h.unsqueeze(2).expand(-1, -1, DEGREE, -1)  # (B,N,6,d)
            nbr, msg = self.message(h)
            if self.no_gate:
                gate = torch.ones(B, NUM_NODES, DEGREE, 1, device=h.device)
            else:
                gate_in = torch.cat([h_center, nbr], dim=-1)
                gate_logits = self.gate_mlp(gate_in)  # (B,N,6,1)
                gate_soft = torch.sigmoid(gate_logits)
                if TOP_K < DEGREE:
                    topk_val, topk_idx = gate_soft.squeeze(-1).topk(TOP_K, dim=-1)
                    hard = torch.zeros_like(gate_soft.squeeze(-1))
                    hard.scatter_(-1, topk_idx, 1.0)
                    gate = (hard - gate_soft.squeeze(-1)).detach() + gate_soft.squeeze(-1)
                    gate = gate.unsqueeze(-1)
                else:
                    gate = gate_soft
            agg = (gate * msg).sum(dim=2) / DEGREE  # (B,N,d)
            h_new = (1 - ALPHA) * (h + K_BASE * agg) + ALPHA * h0
            h_new = self.norms[layer](h_new)
            h = F.gelu(h_new)
        pooled = h.mean(dim=1)  # (B,d) — global average pooling عبر العقد
        return self.readout(pooled)


# --------------------------------------------------------------- البيانات
def get_dataloaders():
    transform = T.Compose([T.ToTensor()])  # 28x28 خام، نصغّر يدوياً بـavg_pool2d
    train_full = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    g = torch.Generator().manual_seed(2026)
    n = len(train_full)
    perm = torch.randperm(n, generator=g)
    val_size = 10_000
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    train_set = Subset(train_full, train_idx.tolist())
    val_set = Subset(train_full, val_idx.tolist())

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])            # (B,1,28,28)
        xs = F.avg_pool2d(xs, kernel_size=2)                # (B,1,14,14) — تجميع متوسط حقيقي، لا استيفاء
        xs = xs.view(xs.size(0), -1)                        # (B,196)
        ys = torch.tensor([b[1] for b in batch])
        return xs, ys

    return (
        DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate),
        DataLoader(val_set, batch_size=512, shuffle=False, collate_fn=collate),
        DataLoader(test_set, batch_size=512, shuffle=False, collate_fn=collate),
    )


# --------------------------------------------------------- تدريب بتقارب حقيقي
@dataclass
class RunResult:
    condition: str
    seed: int
    best_val_acc: float
    best_epoch: int
    test_acc: float
    learning_curve: list = field(default_factory=list)


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def train_one(condition_name, neighbor_idx, positional, model_seed, train_loader, val_loader, test_loader):
    torch.manual_seed(model_seed)
    model = GatedGeomGNN(neighbor_idx.to(DEVICE), positional=positional).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val, best_state, best_epoch, bad_epochs = -1.0, None, -1, 0
    curve = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_acc = evaluate(model, val_loader)
        curve.append(val_acc)
        if val_acc > best_val:
            best_val, best_epoch, bad_epochs = val_acc, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if bad_epochs >= PATIENCE:
            break

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader)
    print(f"  [{condition_name} | seed={model_seed}] best_val={best_val:.4f} @epoch={best_epoch} "
          f"| test={test_acc:.4f} | توقف عند epoch={epoch}")
    return RunResult(condition_name, model_seed, best_val, best_epoch, test_acc, curve)


# --------------------------------------------------------------- التنسيق الرئيسي
def run_condition(name, slot_builder_fn, positional, seeds_key="model"):
    results = []
    if seeds_key == "model":
        neighbor_idx = slots_to_tensor(slot_builder_fn())
        for ms in MODEL_SEEDS:
            results.append(train_one(name, neighbor_idx, positional, ms, *loaders))
    else:  # topology seeds (للرسوم العشوائية)
        for ts in TOPOLOGY_SEEDS:
            neighbor_idx = slots_to_tensor(slot_builder_fn(ts))
            per_topology = [train_one(f"{name}_topo{ts}", neighbor_idx, positional, ms, *loaders)
                             for ms in MODEL_SEEDS[:3]]  # 3 بذور نموذج لكل توبولوجيا (يوازن الكلفة الحسابية)
            results.extend(per_topology)
    return results


def two_level_permutation_test(hex_scores, random_topology_means):
    """اختبار تبديلي: السداسي (عيّنة واحدة) مقابل توزيع متوسطات التوبولوجيات العشوائية."""
    hex_mean = float(np.mean(hex_scores))
    n = len(random_topology_means)
    n_ge = sum(1 for v in random_topology_means if v >= hex_mean)
    p = (1 + n_ge) / (1 + n)
    return hex_mean, p, n


if __name__ == "__main__":
    print(f"الجهاز: {DEVICE}")
    if DEVICE == "cpu":
        print("⚠️ تحذير: بدون GPU هذا التدريب رح ياخذ وقت طويل جداً — شغّله على RTX 3070.")

    hex_slots = build_hex_slots()
    hex_perm_slots = build_hex_perm_slots(hex_slots)

    loaders = get_dataloaders()

    all_results = {}
    _existing = {}
    try:
        with open("results_raw.json", "r", encoding="utf-8-sig") as f:
            _existing = json.load(f)
        if _existing:
            print(f"وجدت نتائج محفوظة سابقاً بـresults_raw.json للشروط: {list(_existing.keys())}")
            print("(بما إن التجربة حتمية بالكامل على هذا الجهاز — نفس البذرة = نفس الرقم — رح نتخطى هذي الشروط ونكمل من بعدها)")
    except FileNotFoundError:
        print("ما فيه results_raw.json سابق — رح نبدأ من الصفر ونحفظ تدريجياً من هلق.")
    except json.JSONDecodeError as e:
        print(f"⚠️ results_raw.json موجود لكن مو JSON صالح ({e}) — رح نبدأ من الصفر ونتجاهله. "
              f"تأكد إنه محفوظ بترميز UTF-8 (بدون BOM لو ممكن) وإن المحتوى JSON صحيح.")

    def save_progress():
        """حفظ تدريجي بعد كل شرط — يحمي من فقدان كل التقدم لو صار انقطاع
        (إعادة تشغيل جهاز، قطع كهرباء...) زي ما صار فعلاً بتشغيلة سابقة."""
        with open("results_raw.json", "w", encoding="utf-8") as f:
            json.dump({k: [r.__dict__ for r in v] for k, v in all_results.items()},
                       f, ensure_ascii=False, indent=2)
        print(f"  💾 تم حفظ التقدم بـresults_raw.json (الشروط المكتملة حتى الآن: {list(all_results.keys())})")

    def run_or_resume(name, fn):
        if name in _existing:
            all_results[name] = [RunResult(**r) for r in _existing[name]]
            print(f"\n== {name} == (متخطّى — موجود مسبقاً بـresults_raw.json)")
        else:
            print(f"\n== {name} ==")
            all_results[name] = fn()
            save_progress()

    run_or_resume("hex_full", lambda: run_condition("hex_full", lambda: hex_slots, positional=True))
    run_or_resume("hex_perm_full", lambda: run_condition("hex_perm_full", lambda: hex_perm_slots, positional=True))
    run_or_resume("local_random_full", lambda: run_condition(
        "local_random_full", build_local_random_slots, positional=True, seeds_key="topology"))
    run_or_resume("global_random_full", lambda: run_condition(
        "global_random_full", build_global_random_slots, positional=True, seeds_key="topology"))

    # ------------------------------------------------------------- التحليل
    def mean_test_acc(results):
        return float(np.mean([r.test_acc for r in results])) * 100

    def topology_group_means(results, n_model_seeds=3):
        vals = [r.test_acc for r in results]
        return [float(np.mean(vals[i:i + n_model_seeds])) * 100
                for i in range(0, len(vals), n_model_seeds)]

    hex_full_acc = mean_test_acc(all_results["hex_full"])
    hex_perm_acc = mean_test_acc(all_results["hex_perm_full"])
    local_means = topology_group_means(all_results["local_random_full"])
    global_means = topology_group_means(all_results["global_random_full"])

    print("\n================= النتائج الخام (٪ على مجموعة الاختبار المحجوزة) =================")
    print(f"hex_full            : {hex_full_acc:.2f}%")
    print(f"hex_perm_full       : {hex_perm_acc:.2f}%  (فرق عن hex_full: {hex_full_acc - hex_perm_acc:+.2f} نقطة)")
    print(f"local_random_full   : {np.mean(local_means):.2f}% ± {np.std(local_means):.2f}  "
          f"(فرق عن hex_full: {hex_full_acc - np.mean(local_means):+.2f} نقطة)")
    print(f"global_random_full  : {np.mean(global_means):.2f}% ± {np.std(global_means):.2f}  "
          f"(فرق عن hex_full: {hex_full_acc - np.mean(global_means):+.2f} نقطة)")

    _, p_local, n_local = two_level_permutation_test([hex_full_acc], local_means)
    _, p_global, n_global = two_level_permutation_test([hex_full_acc], global_means)
    print(f"\np(hex > local_random)  = {p_local:.4f}  (أرضية={1/(1+n_local):.4f}, n_topologies={n_local})")
    print(f"p(hex > global_random) = {p_global:.4f}  (أرضية={1/(1+n_global):.4f}, n_topologies={n_global})")

    print("\n================= القرار (بدون أي عتبة كسرية تعسفية) =================")
    gap_vs_local = hex_full_acc - np.mean(local_means)
    if abs(gap_vs_local) < 3.0:
        verdict = "الفجوة اختفت تقريباً مقابل ضابط المحلية ⇒ المحلية وحدها تفسّر أغلب الفايدة، لا الشكل السداسي تحديداً."
    elif gap_vs_local > 3.0:
        verdict = "فجوة حقيقية باقية حتى بعد عزل المحلية ⇒ فيه شيء إضافي (اتساق الاتجاه/العنقدة) مو مجرد محلية."
    else:
        verdict = "العشوائي المحلي تفوق فعلياً على hex ⇒ نتيجة تستحق تحقيق إضافي، لا تُهمَل."
    print(f"الفجوة (hex_full - local_random_full) = {gap_vs_local:+.2f} نقطة مئوية")
    print(verdict)

    with open("results_raw.json", "w", encoding="utf-8") as f:
        json.dump({k: [r.__dict__ for r in v] for k, v in all_results.items()}, f, ensure_ascii=False, indent=2)
    print("\nكل النتائج الخام + منحنيات التعلّم محفوظة بـ results_raw.json — راجعها بنفسك قبل أي استنتاج.")
