from datetime import datetime
import pandas as pd


def test01():
    dt = datetime(2025, 2, 10, 11, 2, 30)
    print(dt)
    print(dt.date())
    print(dt.time())
    # 获取当前时间
    print(datetime.now())


# 时间戳
def test02():
    # 通过日期字符串生成时间戳日期
    ts = pd.Timestamp('2025-02-10 11:09:30')
    print(ts)

    # 通过时间戳生成日期
    ts = pd.Timestamp(1739156799, unit='s', tz='Asia/Shanghai')
    print(ts)
    ts1 = pd.Timestamp(1739156799111, unit='ms', tz='Asia/Shanghai')
    print(ts1)
    # 错误示例，不添加单位，不能准确转换为日期
    ts2 = pd.Timestamp(1739156811)
    print(ts2)


def test03():
    s = '2025-02-10 11:27:21'
    # 将日期类型的字符串转换为pandas的日期类型
    dt = pd.to_datetime(s)
    print(dt)
    print(type(dt))


def test04():
    # date_range():生成固定时间频率的时间序列
    # 参数：
    # start：开始时间
    # end：结束时间
    # periods：生成时间序列的数量
    # freq：时间频率，D-按天，h-小时，
    dt = pd.date_range('2025-01-01', '2025-01-10', freq='D')
    print(dt)

    dt1 = pd.date_range('2025-02-10', freq='h', periods=24)
    print(dt1)

    dt2 = pd.date_range('2025-02-10 00:00:00', '2025-02-10 23:59:59', freq='h')
    print(dt2)


def test05():
    # Timedelta():表示一个时间段
    # 参数：字符串
    td = pd.Timedelta('1 days 2 hours 3 minutes 4 seconds')
    print(td)

    # 使用参数创建
    td = pd.Timedelta(days=1, hours=3, minutes=4)
    print(td)

    # 使用整数和单位创建，单位：days  hours  minutes  seconds
    td = pd.Timedelta(5, unit='hours')
    print(td)

    dt = datetime.now()
    # Timedelta可以和日期进行加减运算，获取新的日期类型数据
    dt += td
    print(dt)


def test06():
    dt = datetime.now()
    # 日期类型转字符串
    # ds = dt.strftime('%Y-%m-%d %H:%M:%S')
    # ds = dt.strftime('%Y/%m/%d %H:%M:%S')
    ds = dt.strftime('%Y%m%d%H%M%S')
    print(ds)

    # 日期类型的字符串转换为日期类型的对象
    # format的时间格式要和日期字符串的格式一样
    s = '2025-02-10 11:02:30'
    dt1 = datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    print(dt1)


if __name__ == '__main__':
    test06()
