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

![Sourcetree安装步骤1](src\Sourcetree安装步骤1.png)



![Sourcetree安装步骤2](src\Sourcetree安装步骤2.png)



![Sourcetree安装步骤3](src\Sourcetree安装步骤3.png)



![Sourcetree安装步骤4](src\Sourcetree安装步骤4.png)





### **导入SSH**

![导入SSH步骤1](src\导入SSH步骤1.png)



![导入SSH步骤2](src\导入SSH步骤2.png)



![导入SSH步骤3](src\导入SSH步骤3.png)



![导入SSH步骤4](src\导入SSH步骤4.png)



![导入SSH步骤5](src\导入SSH步骤5.png)



![导入SSH步骤6](src\导入SSH步骤6.png)



![导入SSH步骤7](src\导入SSH步骤7.png)



![Plink修改1](src\Plink修改1.png)



![Plink修改2](src\Plink修改2.png)



![Plink修改3](src\Plink修改3.png)



![Plink修改4](src\Plink修改4.png)



![Plink修改5](src\Plink修改5.png)



### 使用

![Sourcetree使用](src\Sourcetree使用.png)

**特别注意：如果是新建的本地仓库，应当在初始化后立刻连接到远程仓库（使用ssh的方式），并且先从远程仓库拉取资源（这样可以保证你的本地仓库与远程仓库的同步，以避免不必要的错误）**



### 补充1：有关额外远程仓库的添加

![Sourcetree使用2](src\Sourcetree使用2.png)



![Sourcetree使用3](src\Sourcetree使用3.png)



![Sourcetree使用4](src\Sourcetree使用4.png)



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







