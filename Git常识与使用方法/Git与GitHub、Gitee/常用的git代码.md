# **常用代码**

**git init**	                                      

​	初始化本地仓库



**git add .**                                           

​	添加所有文件到暂存区

**git add index.html**

​	添加单个文件到暂存区

**git commit -m "提交说明"**           

​	提交到本地仓库

**git push**                                          

​	推送到远程仓库



**git remote add origin https://gitee.com/dema_2/python-basic.git**      

​	连接到远程仓库（以http为例）

**git remote set-url origin https://github.com/lan-2659/python_study.git**     

​	更改连接的远程仓库 （以http为例）

**git clone https://gitee.com/aiajsjjssns/python_study.git**                        

​	克隆远程仓库（以http为例）



**git branch**

​	查看本地仓库的所有分支

**git branch feature-login**

​	创建一个名为 feature-login 的分支

**git branch -d feature-login** 

​	删除名为 feature-login 分支；但是如果这个分支上有未合并的内容，git会拒绝删除

**git branch -D feature-login**

​	强制删除名为 feature-login 分支



# 注意事项

**在使用命令行推送时，必须调用add、commit、push这三条指令**