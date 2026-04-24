# W1 重构建 & 提交指引 · Linux 4060 队友

> **本轮目标**：重 build Docker 镜像 → 验证 → 导出 `LCF@NUDT.tar` → 上传夸克网盘
> **依据 commit**：`2953494` 及之后（data self-heal 已加入 init.sh）
> **时间预算**：约 45 分钟（build 15 min + 验证 10 min + 上传 20 min）
> **W1 提交截止**：2026-04-26 23:59

---

## 0. 为什么需要你重 build

- 你本机 CPU **支持 AVX-512**（今天早些时候已证明能复现 golden MD5 `f13034946...`）
- 主开发机 Linux 5090 **没装 docker**（容器环境，PID 1 不是 systemd）
- Windows 队友本机 docker daemon 也受限
- **所以你的 AVX-512 Linux 是唯一能打出"本地自验一致"的打包节点**

---

## 1. 拉最新代码

```bash
cd ~/THU-BDC2026
git fetch origin
git checkout feat/ensemble-v1
git pull
git log --oneline -3
```

**期望看到**：
```
2953494 fix(docker): data self-heal in init.sh + bundle full data backup
299f492 fix(deps): pin pandas==2.3.3 numpy==2.2.6 to match golden 5090 .venv
...
```

---

## 2. 环境自检（必做，不要跳过）

### 2.1 数据 MD5

```bash
md5sum data/stock_data.csv
# 期望 cf3e0526f3d832b2ea2e3f1dc22c52e9
```

**若不匹配**：从 5090 主机拉 `/root/shared-nvme/w1-data-20260424-final.tar.gz` 解压覆盖：
```bash
tar xzf w1-data-20260424-final.tar.gz   # 覆盖 data/ 和 model_lgb_only/
md5sum data/stock_data.csv              # 再确认
```

### 2.2 模型权重 MD5

```bash
md5sum model_lgb_only/lgb/seed_42_refit/sub_0.txt
# 期望 c517168b73fe4bcb5ecbdb1f1dd2434e
```

### 2.3 CPU 指令集（最关键）

```bash
cat /proc/cpuinfo | grep -m1 flags | tr ' ' '\n' | grep -iE '^avx512' | sort -u
# 期望至少看到 avx512f
```

**若无 avx512**：
- **立刻停止**，发消息给主开发机
- 可能需要换另一台机器，或直接用主机已 save 的 tar（有 AVX-512 差异风险）

### 2.4 Docker 可用性

```bash
docker --version                              # Docker 能跑
docker pull hello-world                       # daemon 网络通
docker pull python:3.12-slim-bookworm         # 基础镜像能拉
```

**若 pull 失败**：daemon 网络问题，配代理或镜像源：
```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf <<EOF
[Service]
Environment="HTTP_PROXY=http://<你的代理>"
Environment="HTTPS_PROXY=http://<你的代理>"
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

---

## 3. Docker Build（15 分钟）

```bash
docker buildx build --platform linux/amd64 -t bdc2026 .
```

**预计进度**：
- `docker.io/python:3.12-slim-bookworm` 拉取：1-2 min
- `ghcr.io/astral-sh/uv` 拉取：30 s
- TA-Lib 下载 + 编译：3-5 min
- `uv pip install -r requirements-submission.txt`：3-5 min
- 其余：1-2 min

**build 成功验证**：
```bash
docker images bdc2026
# 期望 1.6 GB 左右，< 10 GB ✅
```

---

## 4. 双场景 dry-run 验证

### 场景 A · 完整 data（复现 golden MD5）

```bash
rm -rf temp/feature_cache_*
docker compose -f docker-compose.local.yml up
```

**逐项验证**：
```bash
cat output/result.csv
md5sum output/result.csv
```

**期望**：
```
stock_id,weight
300308,0.2
600023,0.2
688187,0.2
688256,0.2
002714,0.2
```
**MD5**: `f13034946c0aaea5cb1e3f2d0d6ad692`

**若不一致**：立刻贴错误 + MD5 给主开发机，**不要继续**。

### 场景 B · 模拟赛方挂载（赛规标准）

赛方按赛规只挂载 `train.csv + test.csv`，测试我们 init.sh 自愈逻辑：

```bash
# 准备模拟挂载目录
rm -rf temp/feature_cache_* /tmp/judge_data
mkdir -p /tmp/judge_data
cp data/train.csv data/test.csv /tmp/judge_data/
ls /tmp/judge_data/
# 必须只有两个文件：train.csv, test.csv
```

```bash
# 跑容器
docker run --rm \
  -v /tmp/judge_data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/temp:/app/temp \
  bdc2026 bash -c "bash /app/init.sh && bash /app/test.sh"
```

**stderr 应看到**：
```
[init] /app/data/ missing required CSVs; self-healing from /app/data_bundled/
[init] merging /app/data/{train,test}.csv -> /app/data/stock_data.csv
[init] merged stock_data.csv: 166190 rows  (数字可能微差)
[init] restoring industry_map.csv from /app/data_bundled/
[init] restoring hs300_history.csv from /app/data_bundled/
[init] restoring csi300_index.csv from /app/data_bundled/
[init] restoring stock_basic.csv from /app/data_bundled/
[init] restoring trade_calendar.csv from /app/data_bundled/
[init] deps OK
[init] init.sh complete
[lgb-only] mcap_constrained Top-5 (min_large=3, cand_k=10, q=0.5)
[lgb-only] target date = 2026-04-23
[lgb-only] wrote ./output/result.csv (5 rows)
```

```bash
cat output/result.csv
md5sum output/result.csv
```

**期望同场景 A**：
- Top-5 == `300308, 600023, 688187, 688256, 002714`
- MD5 == `f13034946c0aaea5cb1e3f2d0d6ad692`

**若 Top-5 一致但 MD5 微差**：检查 train+test 合并是否恰好还原 stock_data.csv（可能有 utf-8 BOM 或列序微差）。Top-5 一致就算通过。

---

## 5. 导出提交 tar（3 分钟）

```bash
docker save -o LCF@NUDT.tar bdc2026:latest
ls -lh LCF@NUDT.tar
md5sum LCF@NUDT.tar
```

**期望**：
- 大小约 1.6-1.8 GB
- 记录 MD5（赛方可能要求报备）

---

## 6. tar 回路验证（模拟赛方 load + run）

删本地镜像，只用 tar 重建：

```bash
docker image rm bdc2026:latest
docker load -i LCF@NUDT.tar
# 期望输出 "Loaded image: bdc2026:latest"

rm -rf temp/feature_cache_*
docker compose -f docker-compose.local.yml up
md5sum output/result.csv
# 必须仍是 f13034946c0aaea5cb1e3f2d0d6ad692
```

**若不一致**：tar 在 save/load 过程中损坏，重新 save。

---

## 7. 回执清单（截图或贴文字给主开发机）

```
=== W1 Build Report · Linux 4060 ===
1. Git commit:     <hash>
2. CPU avx512f:    YES / NO
3. Image size:     <GB>
4. 场景 A MD5:     <md5>   (期望 f13034946c0aaea5cb1e3f2d0d6ad692)
5. 场景 B MD5:     <md5>   (Top-5 匹配即可)
6. tar 大小:       <GB>
7. tar MD5:        <md5>
8. 回路验证 MD5:   <md5>   (期望 f13034946c0aaea5cb1e3f2d0d6ad692)
=========================================
```

---

## 8. 上传夸克网盘

> 赛规明确要求："上传至夸克网盘，并生成对应分享链接，确保永久有效，不要加提取码"

### 8.1 上传

1. 打开夸克网盘（网页版或客户端）
2. 上传 `LCF@NUDT.tar`（1.6 GB，视网速 10-20 分钟）
3. 上传完成后，右键文件 → **分享**

### 8.2 分享设置

- **有效期**：**永久**（赛规必选项）
- **提取码**：**关闭 / 不加**（赛规明确不要加）
- **复制链接**

### 8.3 验证链接

在**隐身浏览器**打开链接，确认：
- 能直接访问（不弹提取码输入框）
- 能下载原 tar
- 下载后 md5sum 一致

### 8.4 提交链接到竞赛平台

登录 https://www.heywhale.com/home/ 的 BDC2026 项目页，提交：
- `result.csv`（你场景 A 跑出的那份，5 行）
- 夸克网盘分享链接

---

## 9. 常见问题与排错

### Q1 · docker pull 超时

```bash
docker info | grep -i proxy   # 看 daemon 代理是否生效
```

### Q2 · 场景 B init.sh 报错 pandas 不支持 utf-8-sig

如果赛方 train.csv 不是 utf-8-sig 编码，init.sh 合并脚本可能报错。
备选方案：不跑场景 B，只跑场景 A（场景 A 足以证明镜像功能）。

### Q3 · tar 文件被网络传输破坏

上传前记 MD5，上传后从网盘下载回来重新 md5sum 对比。

### Q4 · 赛方评测机 CPU 无 AVX-512

赛方是数据中心 Xeon，**几乎一定有 AVX-512**。即使没有，我们的 deterministic_top_k 在大部分情况能吸收差异。这是剩余的 1% 风险。

---

## 10. 时间节点

| 动作 | 预计耗时 | 累计 |
|---|---|---|
| git pull + 自检 | 2 min | 2 min |
| docker build | 15 min | 17 min |
| 场景 A | 3 min | 20 min |
| 场景 B | 3 min | 23 min |
| docker save | 3 min | 26 min |
| tar 回路验证 | 3 min | 29 min |
| 夸克网盘上传 | 15 min | 44 min |
| 填链接提交 | 2 min | 46 min |

**总计约 45 分钟。**

---

## 11. ⚠️ 绝对不要做的事

- ❌ **不要**重训 `bash train.sh`（会改变权重 MD5，破坏 golden state）
- ❌ **不要**改 code/ 下任何文件
- ❌ **不要**重新抓 baostock 数据（stock_data.csv 已锁定为 cf3e0526...）
- ❌ **不要**删 `data_bundled/`（init.sh 自愈依赖）
- ❌ **不要**提交加密码的网盘链接

---

## 12. 若 5 个 MD5 全对 → 提交

- 本机场景 A：`f13034946c0aaea5cb1e3f2d0d6ad692` ✅
- 本机场景 B：（Top-5 一致）✅
- tar save 成功 ✅
- tar load 回路 MD5 == 场景 A ✅
- 网盘下载回的 tar MD5 == 本机 tar MD5 ✅

**五关都过，直接提交链接到平台。**

**任一关不过**：贴错误给主开发机，不要硬提交。

---

*文档版本 · 2026-04-25 · 对应 commit 2953494+*
