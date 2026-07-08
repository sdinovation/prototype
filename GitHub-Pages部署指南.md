# GitHub Pages 部署指南

## 📋 部署步骤

### 第一步：登录 GitHub
1. 打开浏览器，访问 [https://github.com](https://github.com)
2. 如果没有账户，点击 "Sign up" 注册一个免费账户
3. 如果已有账户，点击 "Sign in" 登录

### 第二步：创建新仓库
1. 登录后，点击右上角的 "+" 号，选择 "New repository"
2. 填写仓库信息：
   - **Repository name**: 输入 `username.github.io`（将 `username` 替换为您的 GitHub 用户名）
   - **Description**: 可选，如 "我的个人主页"
   - **Public**: 选择 Public（公开），这是 GitHub Pages 的要求
   - **Add a README file**: 可以勾选
3. 点击 "Create repository" 创建仓库

### 第三步：上传 HTML 文件
1. 进入刚创建的仓库页面
2. 点击 "Add file" → "Upload files"
3. 将 `personal-homepage.html` 文件拖拽到上传区域
4. **重要**：将文件名改为 `index.html`（GitHub Pages 默认查找 index.html）
5. 在 "Commit changes" 区域填写提交信息，如 "添加个人主页"
6. 点击 "Commit changes" 提交

### 第四步：启用 GitHub Pages
1. 在仓库页面，点击 "Settings"（设置）
2. 在左侧菜单中找到 "Pages"
3. 在 "Source" 部分：
   - Branch: 选择 "main" 或 "master"
   - Folder: 选择 "/ (root)"
4. 点击 "Save"

### 第五步：获取您的公开链接
1. 等待 1-2 分钟，GitHub Pages 会自动部署
2. 刷新 Pages 设置页面，您会看到您的网站链接
3. 链接格式为：`https://username.github.io`

---

## ✅ 完成！

部署成功后，您的个人主页链接将是：
```
https://您的GitHub用户名.github.io
```

任何人都可以通过这个链接访问您的个人主页！

---

## 🔧 可选：自定义域名

如果您有自己的域名，可以在 GitHub Pages 设置中添加自定义域名：
1. 在 Pages 设置页面的 "Custom domain" 输入您的域名
2. 在您的域名服务商处配置 DNS 解析

---

## 📝 文件位置

您的个人主页文件位于：
```
g:\demo\personal-homepage.html
```

请将此文件上传到 GitHub 仓库，并重命名为 `index.html`。

---

## ❓ 常见问题

**Q: 为什么文件名要改为 index.html？**
A: GitHub Pages 默认查找 index.html 作为首页，这样访问 `username.github.io` 时会自动显示该页面。

**Q: 部署后页面没有更新？**
A: GitHub Pages 可能需要几分钟来构建和部署，请耐心等待并刷新页面。

**Q: 如何更新内容？**
A: 直接在 GitHub 仓库中编辑或重新上传新的 index.html 文件即可。