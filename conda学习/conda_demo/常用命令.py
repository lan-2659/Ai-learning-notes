"""
conda常用命令

1. 环境管理

    1.1 创建新环境:
        conda create --name myenv python=3.8
    这会创建一个名为 myenv 的新环境，并安装 Python 3.8。

    1.2 激活环境:
        conda activate myenv
    激活名为 myenv 的环境。

    1.3 停用环境:
        conda deactivate
    停用当前环境，返回到默认环境。

    1.4 列出所有环境:
        conda env list
    或者：
        conda info --envs
    显示所有 conda 环境。

    1.5 删除环境:
        conda env remove --name myenv
    删除名为 myenv 的环境。

2. 包管理

    - 安装软件包:
          conda install numpy -c https://pypi.mirrors.ustc.edu.cn/simple/
      安装名为 numpy 的包。
      -c 可选，其后加镜像源

    - 安装特定版本的包:
          conda install numpy=1.18
      安装 numpy 的 1.18 版本。
      可以用这个语句安装或者更换指定版本的包

    - 安装多个包:
          conda install numpy pandas matplotlib
      一次安装多个包。

    - 更新软件包:
          conda update numpy
      更新 numpy 包。
      这个命令语句只能更新包为最新版本，不能更新包为指定版本

    - 更新 conda 本身:
          conda update conda
      这个命令只有位于 base 环境时才能正确运行

    - 卸载包:
          conda remove numpy
      卸载 numpy 包。



3. 查看和搜索

    - 查看所有已安装的包:
          conda list
      列出当前环境中安装的所有包及其版本。

    - 查看指定的已安装的包
          conda list numpy
      列出当前环境中所有'包名'中带有 numpy 的包的信息

    - 搜索包:
          conda search numpy
      搜索包 numpy 的可用版本。



4. 环境配置

    - 导出当前环境配置到 YAML 文件:
          conda env export > environment.yml
      导出当前环境的配置文件（包括安装的所有包和版本）。

    - 从 YAML 文件创建环境:
          conda env create --file environment.yml
      从 environment.yml 文件中创建一个新的环境。

    - 克隆环境:
          conda create --name newenv --clone oldenv
      克隆现有环境 oldenv 为 newenv。



5. 渠道管理

    Conda 频道（channels）是 Conda 包的存储库，用于存储和分发包。

    - 查看当前环境的频道:
          conda config --show channels

    - 添加新的频道:
          conda config --add channels https://conda.anaconda.org/conda-forge

    - 删除频道:
          conda config --remove channels https://conda.anaconda.org/conda-forge



6. 其他有用的命令

    - 检查 conda 配置和信息:
          conda info
      显示有关 conda 环境和配置的详细信息。

    - 清理无用缓存:
          conda clean --all
      删除缓存文件，以释放磁盘空间。

    - 查看 conda 的帮助:
          conda --help



"""