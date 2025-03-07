

response                        # 这个对象一般被称为响应体

response.content # 这是以字节形式（bytes）返回的响应内容。无论响应内容是什么类型（如文本、图片、视频等），response.content 都会返回其原始的字节形式数据。

response.text # 这是以字符串形式返回的响应内容。它会根据响应头部的字符编码自动进行解码或以response.encoding中的方法解码，然后返回字符串。

response.status_code            # 返回响应状态码，只有状态码等于200时请求才成功

response.ok                     # 返回ok属性，如果请求成功 ok = True

response.headers                # 返回响应头信息  

response.encoding = 'utf-8'     # 可以这样为响应内容指定编码格式，这会影响到response.text中返回的内容

response.content.decode('utf-8') # 会返回一个字符串，使用decode()方法可以对字节形式的响应内容进行解码, 可以不指定解码方式，默认为 utf-8

response.url                    # 返回请求网站

response.Content_Tepy           # 返回响应内容的格式