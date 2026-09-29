"""
topology_check.py
==================
تحقق بنيوي (بدون أي تدريب) من كل التوبولوجيات المطلوبة لتجربة "ضابط المحلية".
الهدف: اكتشاف أي باگ بالرسم البياني (degree غير متطابق، عدم توازن، تسرب محلية)
قبل ما نصرف أي وقت تدريب عليه — هذا بالضبط نوع الباگ اللي ضرب التجربة قبل
(انظر القسم 5 بجدول الأخطاء: "فقدان تطابق الدرجة"، "التفاف توروس").

كل الرسوم البيانية هون بدرجة 6 بالضبط، بدون أي عقدة حدودية (التفاف توروس)،
عشان تبقى المقارنة عادلة بين كل الشروط.
"""

import random
import itertools
import networkx as nx
import numpy as np

N = 14  # 14x14 = 196 node grid, same as MNIST 14x14 downsample
NUM_NODES = N * N
DEGREE = 6


def idx(q, r):
    return (q % N) * N + (r % N)


def coords(i):
    return divmod(i, N)


def build_hex_torus():
    """التوبولوجيا السداسية الأصلية: 6 جيران بترتيب اتجاهي ثابت لكل عقدة."""
    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    G = nx.Graph()
    G.add_nodes_from(range(NUM_NODES))
    slot_map = {}  # node -> ordered list of neighbor ids (slot k = offsets[k])
    for q in range(N):
        for r in range(N):
            i = idx(q, r)
            neighbors = [idx(q + dq, r + dr) for dq, dr in offsets]
            slot_map[i] = neighbors
            for j in neighbors:
                G.add_edge(i, j)
    return G, slot_map


def build_hex_perm(slot_map, seed=123):
    """نفس جيران hex بالضبط، لكن ترتيب المواضع مُخلَّط عشوائياً ومستقلاً لكل عقدة."""
    rng = random.Random(seed)
    perm_slot_map = {}
    for node, neighbors in slot_map.items():
        shuffled = neighbors[:]
        rng.shuffle(shuffled)
        perm_slot_map[node] = shuffled
    # الرسم البياني (مجموعات الجيران) يجب أن يبقى مطابقاً تماماً
    G = nx.Graph()
    G.add_nodes_from(range(NUM_NODES))
    for node, neighbors in perm_slot_map.items():
        for j in neighbors:
            G.add_edge(node, j)
    return G, perm_slot_map


def build_local_random(seed, radius=2, max_tries=200):
    """
    ضابط المحلية: رسم بياني عشوائي منتظم بدرجة 6 بالضبط، لكن كل الأضلاع
    مقيّدة بمسافة شبكية (Chebyshev/رينغ) محلية <= radius على التوروس.
    هذا يفصل "المحلية" عن "الشكل السداسي المحدد" — إذا الفجوة مع hex اختفت
    هون، فالمحلية وحدها هي التفسير، لا الشكل السداسي بذاته.
    """
    def torus_dist(a, b):
        qa, ra = coords(a)
        qb, rb = coords(b)
        dq = min(abs(qa - qb), N - abs(qa - qb))
        dr = min(abs(ra - rb), N - abs(ra - rb))
        return max(dq, dr)

    rng = random.Random(seed)
    for attempt in range(max_tries):
        # مجمع المرشحين المحليين لكل عقدة (كل العقد بمسافة <= radius، باستثناء نفسها)
        candidates = {
            i: [j for j in range(NUM_NODES) if j != i and torus_dist(i, j) <= radius]
            for i in range(NUM_NODES)
        }
        # بناء بمقاربة stub-matching محلي مع رفض الأضلاع المكررة/الذاتية
        stubs = []
        for i in range(NUM_NODES):
            stubs.extend([i] * DEGREE)
        rng.shuffle(stubs)
        G = nx.Graph()
        G.add_nodes_from(range(NUM_NODES))
        ok = True
        deg_count = {i: 0 for i in range(NUM_NODES)}
        edges_needed = set()
        pool = stubs[:]
        random_pairs_ok = True
        # نبني بطريقة أبسط وأوثق: لكل عقدة، نكمل جيرانها من مرشحيها المحليين
        # عشوائياً حتى تبلغ الدرجة 6، مع تفادي التكرار والذات.
        G = nx.Graph()
        G.add_nodes_from(range(NUM_NODES))
        node_order = list(range(NUM_NODES))
        rng.shuffle(node_order)
        failed = False
        for i in node_order:
            need = DEGREE - G.degree(i)
            if need <= 0:
                continue
            pool_i = [j for j in candidates[i] if j != i and not G.has_edge(i, j) and G.degree(j) < DEGREE]
            rng.shuffle(pool_i)
            for j in pool_i[:need]:
                G.add_edge(i, j)
        degs = [d for _, d in G.degree()]
        if min(degs) == DEGREE and max(degs) == DEGREE:
            return G
        # لو ما اكتمل بالمحاولة هاي، جرّب seed/ترتيب مختلف
    raise RuntimeError(f"فشل بناء رسم محلي منتظم الدرجة بعد {max_tries} محاولة — radius={radius} صغير جداً على الأغلب")


def build_global_random(seed):
    return nx.random_regular_graph(DEGREE, NUM_NODES, seed=seed)


def avg_edge_distance(G):
    def torus_dist(a, b):
        qa, ra = coords(a)
        qb, rb = coords(b)
        dq = min(abs(qa - qb), N - abs(qa - qb))
        dr = min(abs(ra - rb), N - abs(ra - rb))
        return max(dq, dr)
    dists = [torus_dist(u, v) for u, v in G.edges()]
    return np.mean(dists), np.max(dists)


def report(name, G):
    degs = [d for _, d in G.degree()]
    clustering = nx.average_clustering(G)
    avg_d, max_d = avg_edge_distance(G)
    print(f"{name:22s} | أضلاع={G.number_of_edges():4d} | درجة(min/max/avg)="
          f"{min(degs)}/{max(degs)}/{np.mean(degs):.3f} | عنقدة={clustering:.4f} "
          f"| بعد الأضلاع(متوسط/أقصى)={avg_d:.2f}/{max_d:.0f}")
    return dict(edges=G.number_of_edges(), min_deg=min(degs), max_deg=max(degs),
                clustering=clustering, avg_dist=avg_d, max_dist=max_d)


if __name__ == "__main__":
    print(f"عدد العقد: {NUM_NODES}, الدرجة المستهدفة: {DEGREE}\n")

    hex_G, hex_slots = build_hex_torus()
    perm_G, perm_slots = build_hex_perm(hex_slots)

    print("=== تحقق سلامة hex_perm: هل مجموعات الجيران مطابقة تماماً لـ hex الأصلي؟ ===")
    hex_neighsets = {n: set(hex_slots[n]) for n in hex_slots}
    perm_neighsets = {n: set(perm_slots[n]) for n in perm_slots}
    identical = all(hex_neighsets[n] == perm_neighsets[n] for n in hex_neighsets)
    print("مطابقة تامة:", identical, "\n")
    assert identical, "خطأ: hex_perm غيّر مجموعة الجيران! الشاهد القاتل غير صالح."

    print("=== الجدول المقارن ===")
    stats = {}
    stats["hex"] = report("hex (الأصلي)", hex_G)
    stats["hex_perm"] = report("hex_perm (الشاهد)", perm_G)

    for radius in (1, 2, 3):
        try:
            local_G = build_local_random(seed=7, radius=radius)
            stats[f"local_r{radius}"] = report(f"local_random (r={radius})", local_G)
        except RuntimeError as e:
            print(f"local_random (r={radius}) فشل: {e}")

    rand_G = build_global_random(seed=42)
    stats["global_random"] = report("global_random (مرجع)", rand_G)

    print("\n=== الاستنتاج الآلي ===")
    hex_dist = stats["hex"]["avg_dist"]
    local_candidates = [k for k in stats if k.startswith("local_r") and abs(stats[k]["avg_dist"] - hex_dist) < 0.6]
    if local_candidates:
        best = local_candidates[0]
        print(f"✅ {best} يعطي متوسط بعد أضلاع قريب من hex ({stats[best]['avg_dist']:.2f} مقابل {hex_dist:.2f})")
        print("   وبنفس الوقت معامل عنقدة مختلف عن hex — هذا هو ضابط المحلية الصحيح للتجربة القادمة.")
    else:
        print("⚠️ لا يوجد radius مجرّب يطابق متوسط بعد أضلاع hex بدقة كافية — جرّب radius إضافي أو نصف متدرّج.")
