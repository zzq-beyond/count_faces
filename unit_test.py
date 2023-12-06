import os
from script import Count

class Test(object):

    def __init__(self, count = 0):
        super(Test, self).__init__()
        self.count = count

    #测试能否正常运行
    def test_program(self):
        flag1 = os.system("python script.py ./images")
        if not flag1:
            self.count += 1
    #测试能否通过命令行控制评估指标
    def test_eval(self):
        flag2 = os.system("python script.py ./images evaluation")
        if ((not flag2)):
            self.count += 1
    #测试检测的人脸是否与真实人脸相等
    def test_count(self):
        c = Count()
        if c.count_faces():
            self.count += 1

if __name__ == '__main__':
    t = Test()
    t.test_program()
    t.test_eval()
    t.test_count()
    print("--------------------------")
    print("测试3条,通过{}条, 通过率{}%".format(t.count, (t.count/3)*100))
