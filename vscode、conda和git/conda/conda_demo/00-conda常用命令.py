"""
conda常用命令

1. 环境管理

    - 创建新环境:
        conda create --name myenv python=3.8
        这会创建一个名为 myenv 的新环境，并安装 3.8 版本的 Python 包
        创建环境时可以安装多个包，相邻包之间用空格分割（也可以不安装包）

    - 激活环境:
        conda activate myenv
        激活名为 myenv 的环境。

    - 停用环境:
        conda deactivate
        停用当前环境，返回到默认环境。

    - 列出所有环境:
        conda env list
        按照创建时间，列出所有环境。
        或者：
        conda info --envs
        显示所有 conda 环境。

    - 删除环境:
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
        若待安装的包已存在，这个命令会将原有的包更新为指定版本

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

    - 查看"所有的"已安装包:
        conda list
        列出当前环境中安装的所有包及其版本。

    - 查看"指定的"已安装包
        conda list numpy
        模糊搜索，只要包名中包含 "numpy" 字符串，就会被列出来。

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