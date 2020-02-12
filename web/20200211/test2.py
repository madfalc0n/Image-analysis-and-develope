def renderfile(file, data) :
    html = open(file, "rt", encoding="utf-8").read()# 템플릿의 html 파일을 참조
    for v in data :
        html = html.replace("@"+v, data[v])
    return html

data = {"title":"나의 홈페이지",  "name":"이순신", "email":"^^"}
print(renderfile("template.html", data)) 