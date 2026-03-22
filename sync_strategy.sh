#!/bin/bash

# 1. 获取传入的参数，如果没有传，则使用默认值
COMMIT_MSG=${1:-"Update strategies $(date +'%Y-%m-%d %H:%M:%S')"}

echo "🚀 开始同步私有策略库..."

# 2. 进入 user_data 目录
cd user_data || exit

# 3. 提交 user_data 的改动
git add .
git commit -m "$COMMIT_MSG"
git push origin main

# 4. 返回主目录并更新主仓库对子模块的引用
cd ..
git add user_data
git commit -m "chore: update user_data submodule to -> $COMMIT_MSG"
git push origin develop

echo "✅ 同步完成！当前版本消息: $COMMIT_MSG"

# 5. 远程更新 Oracle 上的仓库
ssh oci "cd quant && sh update_bot.sh"