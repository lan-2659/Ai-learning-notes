# Git/GitHub/Gitee

## **1. Git**

下载地址：https://git-scm.com/downloads/win

- **定义**：Git 是一个开源的分布式版本控制系统（DVCS），用于跟踪文件和目录的变更历史。

- **功能**：
  - **版本控制**：记录代码的修改历史，方便回溯和比较不同版本之间的差异。
  - **分支管理**：支持创建多个分支，方便开发团队并行开发和管理特性、修复等。
  - **本地操作**：每个开发者在本地机器上都有完整的代码仓库副本，可以离线操作，如提交、查看历史等。
  - **合并与冲突解决**：提供强大的合并功能，能够处理多个分支的合并，并解决可能出现的冲突。

- **特点**：
  - **分布式架构**：每个开发者拥有完整的代码仓库副本，便于在本地进行开发。
  - **高效性**：对代码的变更进行高效管理，支持大项目和高频更新。
  - **灵活性**：可以用于各种类型的项目，从个人项目到大型团队项目。

- **常用代码**：

  - git init	                                       初始化本地仓库       

  - git add                                           添加文件到暂存区

  - git commit -m "提交说明"           提交到本地仓库

  - git push                                          推送到远程仓库

  - git remote add origin https://gitee.com/dema_2/python-basic.git      连接到远程仓库（以http为例）

  - git remote set-url origin https://github.com/lan-2659/python_study.git     更改连接的远程仓库 （以http为例）

  - git clone https://gitee.com/aiajsjjssns/python_study.git                        克隆远程仓库（以http为例）

    在使用命令行推送时，必须调用add、commit、push这三条指令



## **2. GitHub**

- **定义**：GitHub 是一个基于 Git 的代码托管平台，由 GitHub, Inc. 提供服务。它在 Git 的基础上增加了许多协作功能，主要用于托管开源项目和私有项目。

- **功能**：

  - **代码托管**：为用户提供代码仓库的托管服务，支持 Git 的所有功能。
  - **协作工具**：提供 Pull Request（拉取请求）、Issue（问题跟踪）、Wiki（文档）等功能，方便团队协作。
  - **社交功能**：支持用户关注、星标（Star）、Fork（分叉）等操作，促进开源社区的交流。
  - **持续集成/持续部署（CI/CD）**：通过 GitHub Actions 等工具支持自动化测试和部署。

- **特点**：

  - **全球影响力**：是目前最大的代码托管平台，拥有庞大的开发者社区。
  - **强大的协作功能**：集成了多种工具和插件，方便团队开发和项目管理。
  - **开源友好**：支持开源项目，促进了开源文化的传播。

  ​

## **3. Gitee**

- **定义**：Gitee（码云）是一个由国内公司（上海奥蓝信息科技有限公司）运营的代码托管平台，类似于 GitHub，但主要面向国内开发者。
- **功能**：

  - **代码托管**：提供基于 Git 的代码托管服务，支持公有和私有仓库。
  - **协作工具**：提供类似于 GitHub 的 Pull Request、Issue、Wiki 等功能。
  - **特色服务**：提供代码托管、项目管理、文档管理、团队协作等功能，还支持一些国内特色的功能，如企业级服务、代码托管与企业内部系统集成等。
- **特点**：

  - **国内优化**：服务器位于国内，访问速度快，更适合国内开发者使用。
  - **企业友好**：提供多种企业级功能，如代码审计、权限管理等。
  - **开源支持**：也支持开源项目，但更注重国内开源生态的建设。




**总结：Git相当于你保存在本地的'日记本'，你可以随时读取或修改其中的内容，并且这个日记本还会保留你的修改记录；而GitHub和Gitee则相当于云端托管平台，可以帮你保管'日记本'，并且支持多人的共同操作**



# SSH

SSH（Secure Shell）是一种用于安全远程登录和管理服务器的网络协议

**主要功能：**

- **安全远程登录**：SSH 允许用户通过网络远程登录到服务器或其他设备，就像直接在本地操作一样。它替代了早期不安全的远程登录协议（如 Telnet）。
- **文件传输**：SSH 支持安全的文件传输，例如通过 SFTP（SSH File Transfer Protocol）或 SCP（Secure Copy Protocol）。
- **端口转发**：SSH 可以将本地端口转发到远程服务器，或者将远程端口转发到本地，实现安全的网络通信。
- **命令执行**：用户可以通过 SSH 在远程服务器上执行命令，而无需登录到完整的会话中。
- **隧道功能**：SSH 可以创建加密的隧道，用于传输其他协议（如 HTTP、FTP 等），确保数据在不安全的网络中传输的安全性。

**SSH 允许本地和远程仓库之间的安全通信，并省去每次推送或拉取代码时输入密码的麻烦**



# GitHub/Gitee SSH配置步骤

**1、配置个人信息**

```
git config --global user.name "你的名字"	# 随便输，这个只会在你进行Git提交时保存
git config --global user.email "你的邮箱"  	# 输入GitHub/Gitee绑定的邮箱
```

**2、生成SSH秘钥**

```
ssh-keygen -t rsa -C "你的邮箱"
```

-t rsa：使用RSA算法生成秘钥

-C：添加备注，通常是你的邮箱地址

密钥(私钥)的默认保存地址：C:\Users\26595\\.ssh\id_rsa

密钥(公钥)的默认保存地址：C:\Users\26595\\.ssh\id_rsa.pub

**3、在你的 GitHub/Gitee 账号中进行公钥和私钥的配置**

**GitHub**

登录你的GitHub账号

点击右上角头像

进入 Settings > SSH and GPG keys > New SSH key 

title 下面的方框可以不用管（相当于输入一个备注），在 key 下面的方框中输入公钥

(可能)进行身份验证

添加成功后会向你的邮箱中发送通知邮件

**Gitee**

登录你的Gitee账号

将鼠标移动到右上角头像旁边的小三角处

在出现的下拉框中选择 账号设置 > SSH公钥

标题下面的方框可以不用管（相当于输入一个备注），在公钥下面的方框中输入公钥

(可能)进行身份验证

添加成功后会向你的邮箱中发送一封邮件

**4、测试连接**

**GitHub**

```
ssh -T git@github.com
```

如果成功会输出类似下面的语句：

Hi lan-2659! You've successfully authenticated, but GitHub does not provide shell access.

**Gitee**

```
ssh -T git@gitee.com
```

如果成功会输出类似下面的语句：

Hi 德玛(@aiajsjjssns)! You've successfully authenticated, but GITEE.COM does not provide shell access.



# **Sourcetree 软件**

### 下载

下载地址：https://www.sourcetreeapp.com/



### 安装：

![Sourcetree安装步骤1](D:\Python：study\Git项目管理软件下载\Sourcetree安装步骤1.png)



![Sourcetree安装步骤2](D:\Python：study\Git项目管理软件下载\Sourcetree安装步骤2.png)



![Sourcetree安装步骤3](D:\Python：study\Git项目管理软件下载\Sourcetree安装步骤3.png)



![Sourcetree安装步骤4](D:\Python：study\Git项目管理软件下载\Sourcetree安装步骤4.png)





### **导入SSH**

![导入SSH步骤1](D:\Python：study\Git项目管理软件下载\导入SSH步骤1.png)



![导入SSH步骤2](D:\Python：study\Git项目管理软件下载\导入SSH步骤2.png)



![导入SSH步骤3](D:\Python：study\Git项目管理软件下载\导入SSH步骤3.png)



![导入SSH步骤4](D:\Python：study\Git项目管理软件下载\导入SSH步骤4.png)



![导入SSH步骤5](D:\Python：study\Git项目管理软件下载\导入SSH步骤5.png)



![导入SSH步骤6](D:\Python：study\Git项目管理软件下载\导入SSH步骤6.png)



![导入SSH步骤7](D:\Python：study\Git项目管理软件下载\导入SSH步骤7.png)



![Plink修改1](D:\Python：study\Git项目管理软件下载\Plink修改1.png)



![Plink修改2](D:\Python：study\Git项目管理软件下载\Plink修改2.png)



![Plink修改3](D:\Python：study\Git项目管理软件下载\Plink修改3.png)



![Plink修改4](D:\Python：study\Git项目管理软件下载\Plink修改4.png)



![Plink修改5](D:\Python：study\Git项目管理软件下载\Plink修改5.png)



### 使用

![Sourcetree使用](D:\Python：study\Git项目管理软件下载\Sourcetree使用.png)

**特别注意：如果是新建的本地仓库，应当在初始化后立刻连接到远程仓库（使用ssh的方式），并且先从远程仓库拉取资源（这样可以保证你的本地仓库与远程仓库的同步，以避免不必要的错误）**



### 补充1：有关额外远程仓库的添加

![Sourcetree使用2](D:\Python：study\Git项目管理软件下载\Sourcetree使用2.png)



![Sourcetree使用3](D:\Python：study\Git项目管理软件下载\Sourcetree使用3.png)



![Sourcetree使用4](D:\Python：study\Git项目管理软件下载\Sourcetree使用4.png)



### 补充2：在 SourceTree 上集成 Git LFS

Git LFS 是 GitHub 提供的用来管理大文件的工具。LFS 会将大文件存储在单独的服务器上，而不是直接存储在 Git 仓库中。

**下载与安装**

访问 [Git LFS 官网](https://git-lfs.github.com/) 下载适合你操作系统的版本

用下面这个代码验证安装是否成功

```
git lfs install
```

**在 SourceTree 中启用 Git LFS**

1. 打开 SourceTree。
2. 确保你的仓库已加载到 SourceTree 中。
3. 在 SourceTree 的顶部菜单中，点击 **工具 -> 选项**。
4. 在 **Git** 选项卡中，确保 Git LFS 已启用：
   - 勾选 **启用 Git LFS**。
   - 点击 **确定** 保存设置。

**标记大文件**

1. 在 SourceTree 中，打开你的仓库。

2. 点击顶部菜单的 **命令行模式** 按钮，打开内置终端。

3. 在终端中运行以下命令，标记大文件：

   - 标记单个文件：

     ```
     git lfs track "file's path"
     ```

   - 标记某种类型的文件（例如所有 `.exe` 文件）：

     ```
     git lfs track "*.exe"
     ```

   - 标记某种目录下的文件（例如 `data/` 目录下的所有文件）：

     ```
     git lfs track "data/*"
     ```

4. 运行以下命令，查看已标记的文件：

   ```
   git lfs track
   ```

5. 提交 `.gitattributes` 文件：

   - Git LFS 会生成一个 `.gitattributes` 文件，记录哪些文件需要被 LFS 跟踪。
   - 在 SourceTree 中，你会看到 `.gitattributes` 文件被修改。
   - 勾选 `.gitattributes` 文件，填写提交信息，然后点击 提交。

**重新添加大文件**

1. 在 SourceTree 的文件状态面板中，你会看到大文件已被标记为需要 LFS 跟踪。
2. 勾选这些大文件，填写提交信息，然后点击 提交。

**推送到远程仓库**

1. 在 SourceTree 中，点击 推送 按钮。
2. 选择要推送的分支（例如 `master`），然后点击 确定。
3. SourceTree 会将大文件通过 Git LFS 上传，并将指针文件推送到 GitHub。







