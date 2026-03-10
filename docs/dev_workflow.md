# Git 开发模板（v1 基线版）

该仓库已将 `main` 作为第一版基线，后续改动按以下流程提交。

## 分支约定

- `main`：发布分支（只放经过验证的稳定代码）
- `fix/<name>`：小修（bug、日志、参数）
- `feat/<name>`：新功能（新逻辑、实验功能）
- `chore/<name>`：工程类变更（格式、依赖、脚本）

## 小修建议流程

```bash
git checkout main
git pull --ff-only origin main
git checkout -b fix/<name>

# 修改文件
python -m compileall -q rna_backbone_design train_se3_flows.py
git add -u rna_backbone_design train_se3_flows.py
git commit -m "fix: <描述>"

git checkout main
git merge --ff-only --no-edit fix/<name>   # 或 --no-ff
git push origin main
```

## 新功能建议流程

```bash
git checkout main
git pull --ff-only origin main
git checkout -b feat/<name>

# 开发 + 验证
python -m compileall -q rna_backbone_design train_se3_flows.py

git add .
git commit -m "feat: <描述>"

git checkout main
git merge --no-ff -m "feat: <name>" feat/<name>
git push origin main
```

## 回滚策略

- 本地提交后撤回：`git reset --soft HEAD~1` / `git reset --hard HEAD~1`
- 已推送提交撤回：`git revert <commit_id>`
- 如需废弃远端错误提交，先评估影响再执行 `git push --force-with-lease origin main`

## 发布检查清单（每次合并前）

- [ ] 目标分支是 `main`
- [ ] 只含当前要发布的更改（无无关实验）
- [ ] 关键文件已编译通过
- [ ] 代码能在单卡和 `main` 的 `CUDA` 配置下启动一次
- [ ] commit message 清晰，描述包含“动机 + 影响文件”
