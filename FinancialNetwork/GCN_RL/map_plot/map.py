# -*- coding:utf-8 -*-
# 2022-2-14
# 作者：小蓝枣
# pyecharts地图

# 需要引用的库
from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.render import make_snapshot
from pyecharts.commons.utils import JsCode

# 设置国家及其位置标识
countries = [
   "China", "Japan", "Italy", "United Kingdom",
    "France", "Austria", "Germany"
]

# 图片路径
year = "2022"
ratio = "train0.6_val0.15_test0.25"


def create_world_map():
    '''
     作用：生成世界地图并保存为图片
    '''
    world_map = (
        Map()
        .add(
            series_name="国家标识",
            data_pair=[[country, 1] for country in countries],  # 标识国家
            maptype="world",
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="世界地图"),
            visualmap_opts=opts.VisualMapOpts(max_=1, is_piecewise=False),
        )
        .set_series_opts(
            label_opts=opts.LabelOpts(is_show=False)
        )
    )

    # 添加图片显示功能
    for country in countries:
        img_path = f'../vision_dataset/img/{country}/{year}/{ratio}/分类结果.png'
        js_code = JsCode(f"""
            function (params) {{
                if (params.name === '{country}') {{
                    var img = new Image();
                    img.src = '{img_path}';
                    img.style.width = '200px';
                    img.style.height = '200px';
                    img.style.position = 'absolute';
                    img.style.top = '100px';
                    img.style.left = '100px';
                    document.body.appendChild(img);

                    setTimeout(function () {{
                        document.body.removeChild(img);
                    }}, 2000);
                }}
            }}
        """)
        world_map.add_js_funcs(
            f"""
            chart.on('mouseover', {js_code});
            """
        )

    # 渲染为HTML文件
    world_map.render("世界地图1.html")




create_world_map()
