"""Kiểm tra tự viết cho các phần dễ sai (RUBRIC mục C và H). Chạy không cần GPU, không cần ảnh:

    cd submissions/2A202602507_pham_van_hoang_anh_tu/code && python -m unittest test_code -v
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import losses  # noqa: E402
import model as M  # noqa: E402
import train  # noqa: E402


class TestLosses(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.logits = torch.randn(64, 9, generator=g) * 3
        self.y = torch.randint(0, 9, (64,), generator=g)

    def test_focal_gamma0_equals_ce(self):
        fl = losses.FocalLoss(gamma=0.0)(self.logits, self.y)
        self.assertLess(abs(fl.item() - F.cross_entropy(self.logits, self.y).item()), 1e-6)

    def test_focal_downweights_easy_examples(self):
        self.assertLess(losses.FocalLoss(2.0)(self.logits, self.y).item(),
                        F.cross_entropy(self.logits, self.y).item())

    def test_label_smoothing(self):
        ce = F.cross_entropy(self.logits, self.y).item()
        self.assertLess(abs(losses.LabelSmoothingCE(0.0)(self.logits, self.y).item() - ce), 1e-6)
        ref = F.cross_entropy(self.logits, self.y, label_smoothing=0.1).item()
        self.assertLess(abs(losses.LabelSmoothingCE(0.1)(self.logits, self.y).item() - ref), 1e-6)

    def test_class_weights(self):
        counts = [675, 637, 618, 613, 637, 605, 644, 609, 5463]  # train fold 0
        w = losses.class_weights(counts)
        self.assertAlmostEqual(w.mean().item(), 1.0, places=5)
        self.assertLess(w[8].item(), w[0].item())  # Negatives được trọng số nhỏ nhất
        wb = losses.class_weights(counts, beta=0.999)
        self.assertAlmostEqual(wb.sum().item(), 9.0, places=4)

    def test_cutmix_lam_matches_real_area(self):
        x = torch.zeros(8, 3, 32, 32)
        x[4:] = 1.0  # nửa sau batch toàn số 1 để đo diện tích dán
        for seed in range(20):
            xm, (ya, yb, lam) = losses.mix_batch(x, torch.arange(8), 1.0, "cutmix", rng=np.random.default_rng(seed))
            perm = yb
            for i in range(8):
                if (x[i, 0, 0, 0] == 0) and (x[perm[i], 0, 0, 0] == 1):
                    pasted = xm[i, 0].mean().item()  # tỉ lệ điểm ảnh lấy từ ảnh kia
                    self.assertAlmostEqual(1 - lam, pasted, places=6)

    def test_mixup_and_mixed_loss(self):
        x = torch.randn(4, 3, 8, 8)
        y = torch.tensor([0, 1, 2, 3])
        xm, (ya, yb, lam) = losses.mix_batch(x, y, 0.4, "mixup", rng=np.random.default_rng(1))
        torch.testing.assert_close(xm, lam * x + (1 - lam) * x[yb])  # y = arange nên yb chính là perm
        ce = torch.nn.CrossEntropyLoss()
        lg = torch.randn(4, 9)
        self.assertAlmostEqual(losses.mixed_loss(ce, lg, (ya, yb, lam)).item(),
                               lam * ce(lg, ya).item() + (1 - lam) * ce(lg, yb).item(), places=5)


class TestModelAndOptim(unittest.TestCase):
    def setUp(self):
        self.m = M.build_model("resnet18", pretrained=False)

    def test_param_groups_no_decay_on_norm_bias(self):
        groups = {g["name"]: g for g in M.param_groups(self.m, 1e-4, 1e-3, 0.05)}
        for g in groups.values():
            if g["name"].endswith("no_decay"):
                self.assertEqual(g["weight_decay"], 0.0)
                self.assertTrue(all(p.ndim <= 1 for p in g["params"]))
        self.assertEqual(groups["head_decay"]["lr"], 1e-3)
        self.assertEqual(groups["backbone_decay"]["lr"], 1e-4)
        n = sum(p.numel() for g in groups.values() for p in g["params"])
        self.assertEqual(n, sum(p.numel() for p in self.m.parameters()))

    def test_frozen_only_head_trains_and_bn_stays_eval(self):
        m = M.build_model("resnet18", pretrained=False, init="frozen")
        groups = M.param_groups(m, 1e-4, 1e-3, 0.05)
        head_ids = {id(p) for p in m.get_classifier().parameters()}
        self.assertTrue(all(id(p) in head_ids for g in groups for p in g["params"]))
        M.set_train_mode(m)
        bns = [x for x in m.modules() if isinstance(x, torch.nn.BatchNorm2d)]
        self.assertTrue(bns and all(not b.training for b in bns))
        before = bns[0].running_mean.clone()
        m(torch.randn(4, 3, 64, 64))
        torch.testing.assert_close(bns[0].running_mean, before)

    def test_count_params_and_gmacs(self):
        self.assertAlmostEqual(M.count_params(self.m), 11.18, places=1)
        self.assertAlmostEqual(M.count_gmacs(self.m, 224), 1.82, delta=0.1)  # ResNet-18 ~1,8 GMAC

    def test_scheduler_warmup_then_cosine(self):
        f = [train.lr_factor(s, 100, 10) for s in range(100)]
        self.assertAlmostEqual(f[0], 0.1)
        self.assertAlmostEqual(f[9], 1.0)
        self.assertTrue(all(a >= b for a, b in zip(f[10:], f[11:])))
        self.assertLess(f[-1], 0.01)

    def test_ema(self):
        net = torch.nn.Linear(3, 2)
        ema = train.EMA(net, 0.9)
        w0 = net.weight.detach().clone()
        with torch.no_grad():
            net.weight.add_(1.0)
        ema.update(net)
        torch.testing.assert_close(ema.module.weight, 0.9 * w0 + 0.1 * (w0 + 1.0))

    def test_parse_overrides(self):
        d = train.parse_overrides(["seed=1", "loss=focal", "ema_decay=none", "amp=false", "lr_head=3e-3"])
        self.assertEqual(d, {"seed": 1, "loss": "focal", "ema_decay": None, "amp": False, "lr_head": 3e-3})
        with self.assertRaises(KeyError):
            train.parse_overrides(["khong_co=1"])

    def test_defaults_match_guide(self):
        c = train.Config()
        self.assertEqual((c.epochs, c.batch_size, c.lr_backbone, c.lr_head, c.weight_decay),
                         (12, 64, 1e-4, 1e-3, 0.05))
        self.assertFalse(c.save_test_predictions)


class TestInference(unittest.TestCase):
    """Bước 3: TTA, gộp view, temperature scaling, soup, gộp BN, đo độ trễ."""

    def test_hflip_and_multicrop(self):
        import inference as inf
        x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
        self.assertTrue(torch.equal(inf.view_hflip(x)[..., 0], x[..., -1]))
        crops = inf.views_multicrop(x, 6, flip=True)
        self.assertEqual(len(crops), 10)
        self.assertTrue(all(c.shape == (2, 3, 6, 6) for c in crops))
        self.assertTrue(torch.equal(crops[0], x[..., :6, :6]))

    def test_aggregate_prob_vs_logit(self):
        import inference as inf
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=(5, 9)), rng.normal(size=(5, 9))
        for space in ("prob", "logit"):
            p = inf.aggregate_views([a, b], space)
            np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-9)
        np.testing.assert_allclose(inf.aggregate_views([a], "prob"), inf.aggregate_views([a], "logit"), atol=1e-12)

    def test_temperature_recovers_true_T(self):
        import inference as inf
        rng = np.random.default_rng(1)
        z = rng.normal(scale=3.0, size=(4000, 9))
        p = inf.apply_temperature(z, 2.5)
        y = np.array([rng.choice(9, p=row) for row in p])
        T = inf.fit_temperature(z * 1.0, y)  # dữ liệu sinh với T = 2,5
        self.assertAlmostEqual(T, 2.5, delta=0.25)
        self.assertAlmostEqual(inf.fit_temperature_views([z], y, "logit"), T, places=3)

    def test_uniform_soup_is_mean(self):
        import inference as inf
        a = {"w": torch.ones(3), "n": torch.tensor(5)}
        b = {"w": torch.zeros(3), "n": torch.tensor(7)}
        s = inf.uniform_soup([a, b])
        self.assertTrue(torch.allclose(s["w"], torch.full((3,), 0.5)))
        self.assertEqual(int(s["n"]), 5)

    def test_fuse_conv_bn_exact(self):
        import inference as inf
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.BatchNorm2d(8), torch.nn.ReLU(),
                                  torch.nn.Conv2d(8, 4, 1, bias=False), torch.nn.BatchNorm2d(4))
        net.train()
        for _ in range(3):
            net(torch.randn(16, 3, 10, 10))  # cập nhật running stats khác mặc định
        net.eval()
        fused = inf.fuse_conv_bn(net)
        self.assertEqual(fused.n_fused, 2)
        x = torch.randn(4, 3, 10, 10)
        with torch.no_grad():
            self.assertLess(float((net(x) - fused(x)).abs().max()), 1e-5)

    def test_bench_percentiles(self):
        import benchmark
        r = benchmark.bench(lambda: sum(range(1000)), warmup=3, iters=50)
        self.assertEqual(r["n"], 50)
        self.assertLessEqual(r["p50"], r["p95"])
        self.assertLessEqual(r["p95"], r["p99"])


if __name__ == "__main__":
    unittest.main()
