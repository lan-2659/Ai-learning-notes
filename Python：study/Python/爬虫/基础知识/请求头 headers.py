
# 每用到一个参数，我会在此处保存一个参数


# 请求头
'''
headers = { 
    'User-Agent': '标识发送请求的客户端信息, 这会告诉网址这个请求来自哪里'  # 可以通过对它的修改伪装请求来源
    'Cookie': '记录用户的活动、存储用户的偏好设置，以及为网站提供其他有关用户的信息(如登录信息)' # 如果网站需要登陆，把它带上就好
    }
'''
# 注意 Cookie 是保存在客户端浏览器中的，可以被一些恶意程序或者跨站脚本（XSS）攻击获取和操纵

# user-agent池的构造方法
'''
第一种：(手打)
    ua_list = [
        'User-Agent':'第一个地址',
        'User-Agent':'第二个地址',
        'User-Agent':'第三个地址'
        ···
        ]

第二种：(使用 fake_useragent 库中的 UserAgent 类的 random 方法)(可能会出现异常)
    from fake_useragent import UserAgent
    ua_list = [
        {'User-Agent':f'{UserAgent().random}'},
        {'User-Agent':f'{UserAgent().random}'},
        {'User-Agent':f'{UserAgent().random}'}
        ···
        ] 
    # 注意使用 random 方法时不要带 () ,UserAgent 类的 random 方法后没有()

使用User-Agent池的方法：
    import random
    headers = {}
    headers.update(random.choice(ua_list))

'''