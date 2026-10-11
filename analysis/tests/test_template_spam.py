import numpy as np
import pandas as pd

from kma import template_spam as ts

FRAME = (
    "Katika Kenya tumia leo msimbo wa kuponi temu kwa oda ya kwanza misimbo ya promo "
    "kuponi za punguzo voucher mikopo bure punguzo promocode vouchers"
)


def coupon_posts(n: int = 8) -> pd.DataFrame:
    fillers = ["The athlete flies", "loudly He eats", "Our dog jumps", "A scientist writes",
               "She eagerly reads", "The teacher paints", "My brother runs", "The engineer eats"]
    emoji = ["\U0001f406", "\U0001faa9", "\U0001f600", "\U0001f9f5", "\U0001f3ef", "\U0001f604",
             "\U0001f969", "\U0001f9ff"]
    rows = []
    for k in range(n):
        filler = fillers[k % len(fillers)].replace(" ", emoji[k % len(emoji)], 1)
        rows.append({"user_id": f"u{k}", "text": f"{filler} to the park {FRAME}‍ https://t.co/x{k}"})
    return pd.DataFrame(rows)


def organic_posts() -> pd.DataFrame:
    texts = [
        "the governor race in nairobi is wide open this time",
        "polls show sifuna ahead but undecided voters are many",
        "why does nobody talk about the fuel levy anymore",
        "county budgets passed without any public participation again",
        "new bypass road will cut travel time to thika by half",
        "farmers in nakuru demand a fair price for milk",
    ]
    return pd.DataFrame({"user_id": [f"o{k}" for k in range(len(texts))], "text": texts})


def test_normalise_folds_variation():
    assert ts.normalise("kuponi ᵗᵉᵐᵘ \U0001f406 http://x.co @a") == "kuponi temu"
    assert ts.normalise("pro​mo c0up0n") == "promo coupon"
    assert ts.normalise("Pay 2026 now!!") == "pay 2026 now"
    assert ts.normalise(None) == ""


def test_coupon_template_is_flagged_as_promo_template():
    stats = ts.template_stats(coupon_posts())
    assert stats.template_rate == 1.0
    assert stats.promo_rate == 1.0
    assert stats.link_rate == 1.0
    assert ts.classify(stats) == "template_promo"


def test_organic_same_topic_is_not_flagged():
    stats = ts.template_stats(organic_posts())
    assert stats.template_rate == 0.0
    assert ts.classify(stats) is None


def test_template_without_promo_vocabulary_is_template_only():
    frame = "shame on john mutunga ndindi nyoro kibuuline glispa bomas convention centre wallabies"
    posts = pd.DataFrame({"user_id": [f"u{k}" for k in range(5)],
                          "text": [f"{frame} #KenyaDecides"] * 5})
    stats = ts.template_stats(posts)
    assert ts.classify(stats) == "template"


def test_copies_by_one_account_do_not_make_a_frame():
    posts = pd.DataFrame({"user_id": ["same"] * 6, "text": [FRAME] * 6})
    stats = ts.template_stats(posts)
    assert stats.template_rate == 0.0
    assert ts.classify(stats) is None


def test_too_few_posts_is_never_flagged():
    stats = ts.template_stats(coupon_posts(3))
    assert stats.eligible == 3
    assert ts.classify(stats) is None


def test_short_posts_are_not_eligible():
    shares = ts.frame_shares(["hi there", "ok"], ["a", "b"])
    assert np.isnan(shares).all()


def test_cap_is_deterministic():
    big = pd.concat([coupon_posts(8)] * 2, ignore_index=True)
    a = ts.template_stats(big, cap=10, seed=1)
    b = ts.template_stats(big, cap=10, seed=1)
    assert a == b and a.posts == 10
