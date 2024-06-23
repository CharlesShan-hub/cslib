# ''' Test `lab` Loading
# '''
# import sys
# import os
# sys.path.append(os.getcwd())
# import libs.data
# from libs.utils import path_load_test
# import libs
# # This will print Hello World!
# path_load_test() # <- open this comment
# print(libs)


# 计算上一段英文文章的字数
essay = """
Balancing Present and Future

The dilemma of whether to prioritize the present or plan for the future is a complex one. While there is an undeniable charm to living in the moment, it is argued that strategizing for the future is essential for enduring success and contentment.

First and foremost, the act of establishing objectives plays a pivotal role in fostering growth. Possessing a clear-eyed perspective on the future serves as a guiding light, steering choices and endeavors, and instilling within individuals a profound sense of direction. This forward-thinking methodology not only propels personal advancement but also paves the way for the realization of deepest ambitions.

Furthermore, life’s unpredictability necessitates preparation. By readying oneself for the trials that lie ahead—be it through the creation of financial safety nets or the cultivation of problem-solving prowess—resilience is fortified. This foresight equips individuals to confront the unknown with poise and assurance.

Additionally, serendipity tends to favor those who are well-prepared. Whether in the realms of academia, professional endeavors, or entrepreneurial pursuits, meticulous advance planning sharpens the capacity to discern and capitalize on potential chances, thereby bolstering prospects for triumph.

Nonetheless, it is equally imperative to immerse oneself in the richness of the present. Engaging in activities that breed joy and nurturing meaningful relationships imbue lives with immediate happiness and contribute to a well-rounded equilibrium between labor and leisure.

In conclusion, while relishing the now is indeed essential, so too is the art of future planning. Achieving this equilibrium is the linchpin of a life that is both meaningful and prosperous.
"""
print(len(essay.replace("\n", " ").replace("  ", " ").split(" "))
)