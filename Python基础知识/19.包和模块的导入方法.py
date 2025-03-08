
# 创建模块
'''
模块是扩展名为.py的文件
如果在同一目录下可以直接导入该模块
导入该模块后可以调用该模块的方法

导入整个模块后使用该模块 方法/类 格式:
模块名.方法名(参数)
模块名.类名(参数)
'''



'''
导入模块时会将模块中的代码运行一遍
可以用 if __name__ == __main__: 语句来限制一些语句的运行，比如测试语句

__file__  绑定模块的路径
__name__  绑定模块的名称
       如果是主模块（首先启动的模块）则绑定 '__main__'
       如果不是主模块则 绑定 xxx.py 中的 xxx 这个模块名
'''



# 导入模块
'''
格式:
import 模块名          # 导入整个模块，用 模块名.方法名/类名/属性名 这个语句调用模块中的内容

form 模块名 import 类名1/方法1, 类名2/方法2, 类名3/方法3, ……      # 通过这个方法可以导入指定的模块或方法(属性也可以导入，懒得写)
form 模块名 import *       #使用*导入全部东西，但不推荐用
    这两种导入方法，可以直接用 方法名/类名/属性名 调用了,再用原来方式调用方法会报错
'''


# 为导入 函数/模块/属性 或 模块本身 起别名
'''
from 模块名 import 方法名/类名/属性名 as 小名 
import 模块名 as 小名
# 用了as起别名后原名就不能用了, 一用就报错
'''


# 同模块的导入规则
'''
import 包名 [ as 包别名]
import 包名.模块名 [ as 模块新名]
import 包名.子包名.模块名 [ as 模块新名]

from 包名 import 模块名 [ as 模块新名]
from 包名.子包名 import 模块名 [ as 模块新名]
from 包名.子包名.模块名 import 属性名 [ as 属性新名]

# 导入包内的所有子包和模块
from 包名 import *
from 包名.模块名 import *
'''


# 相对导入规则
'''
只有属于同一个包时才可以用
. 代表当前包或当前目录。
.. 代表父包或父目录。
... 代表祖父包，依此类推。
'''


# 第三方包
'''
pip install package-name            # 在命令行中运行这个命令来安装包

pip install package-name==version   # 安装特定版本的包

pip install package-name -i https://pypi.tuna.tsinghua.edu.cn/simple 
    # 使用 -i 参数指定下载源地址 （网上搜镜像源，一搜一大堆）
    
pip install -r requirements.txt  # 可以将要安装的包及其版本记录在一个文本文件中，通常命名为requirements.txt

pip install --upgrade package-name  # 更新包

pip uninstall package-name      # 卸载包
'''


# pipreqs包
'''
pipreqs是一个用于管理Python项目依赖清单的工具

进入你的项目目录，然后运行以下命令：
    pipreqs .
这会分析项目代码，并在当前目录下生成一个名为requirements.txt的文件
其中包含了项目所需的所有依赖包及其版本。

不过，需要记住，生成的依赖清单可能包含一些不必要的依赖，
因此你应该仔细检查和编辑requirements.txt文件以确保它反映了项目的真实需求。
'''










