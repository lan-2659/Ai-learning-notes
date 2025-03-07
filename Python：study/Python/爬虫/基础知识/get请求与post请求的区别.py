

# 使用方面
'''
get请求 -- 使用比较多
post请求 -- 使用比较少
'''


# 请求方法
'''
get请求会直接向服务器发送请求，获取响应内容
post请求会先给服务器一些参数，再获取响应内容
'''


# 携带参数
# get请求中没有data参数，post请求中有 params 参数但不推荐使用
'''
get请求 -- params参数
post请求 -- data参数

注意：
get请求会在 url 后面附加上 params 中的参数
post不会在 url 后面附加上 data 中参数
'''