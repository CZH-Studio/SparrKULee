from typing import List

import requests
from lxml import html
import json

def main():
    # 获取所有数据文件的url
    base_url = "https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/K3VSND"
    headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,zh-TW;q=0.5",
            "Connection": "keep-alive",
            "Cookie": "JSESSIONID=6976a5206fe56aeddcf060b7a5d4",
            "Host": "rdr.kuleuven.be",
            "Sec-Ch-Ua": '"Microsoft Edge";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "same-origin",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
        }
    response = requests.get(base_url, headers=headers, timeout=10)
    response.raise_for_status()
    content = response.text
    tree = html.fromstring(content.encode("utf-8"))
    json_string = tree.xpath("//script/text()")[0]
    json_dict = json.loads(json_string)
    datafile_urls: List[str] = [item["contentUrl"] for item in json_dict["distribution"]]
    with open("./dataset_link.txt", 'w', encoding="utf-8") as f:
        for url in datafile_urls:
            f.write(url + '\n')


if __name__ == '__main__':
    main()