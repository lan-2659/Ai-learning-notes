def check_password_strength(password: str) -> str:
    length = len(password)
    response = ''
    match length: # 匹配长度
        case 0:
            response += '无效密码'
        case length if length < 6:
            response += '弱密码（长度过短）'
        case length if 6 <= length < 10:
            response += '中等强度密码'
        case length if length >= 10:
            response += '强密码（长度合格）'
        
    match length: # 特殊匹配
        case length if length > 8 and any(c in '!@#$%^&*' for c in password) and any(c.isdigit() for c in password):
            response = "非常强的密码！"
        case length if password.upper() == password or password.lower() == password:
            pass
        case length if any(c.isupper() for c in password) and any(c.islower() for c in password):
            response += "（包含大小写混合）"
 
    return response

def process_tuple(data):
    match data:
        case (_, 10, _):  # 匹配第二个元素为 10 的三元组
            return f"中间值为 10: {data}"
        case (xkj, y, z):  # 匹配任意三元组（上面的分支已排除中间值为 10 的情况）
            return f"其他三元组: {data}"
        case _:  # 匹配非三元组的情况
            return "不是三元组"
        
def match_data(data):
    match data:
        case (_, _, _):
            print("这是一个三元组")
    
    match data:
        case (z, x, y):
            print(f"这是一个三元组：{z, x, y}")


if __name__ == "__main__":
    # pass
    match_data((1, 2, 3)) if 1+2 == 4 else print("匹配失败")