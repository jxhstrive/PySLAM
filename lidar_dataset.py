# -*- coding=utf-8


import sys
import os
import re
import logging
import argparse
import multiprocessing

from qcloud_cos import CosConfig, CosServiceError
from qcloud_cos import CosS3Client

# 正常情况日志级别使用 INFO，需要定位时可以修改为 DEBUG，此时 SDK 会打印和服务端的通信信息
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

class cloudData():
    def __init__(self, cfg_path, type, bucket_name, prefix):
        self.bucket_name = bucket_name
        self.prefix = prefix

        import yaml
        with open(cfg_path) as f:
            self.yaml = yaml.safe_load(f)  # sensor dict
        secret_id = self.yaml[type]["secret_id"]
        secret_key = self.yaml[type]["secret_key"]
        region = self.yaml[type]["region"]
        token = None  # 如果使用永久密钥不需要填入 token，如果使用临时密钥需要填入，临时密钥生成和使用指引参见 https://cloud.tencent.com/document/product/436/14048
        scheme = 'https'  # 指定使用 http/https 协议来访问 COS，默认为 https，可不填

        config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token,
                           Scheme=scheme)  # 获取配置对象
        self.client = CosS3Client(config)

    # 列出当前目录子节点，返回所有子节点信息
    def list_currentDir(self, delimiter=''):
        file_infos = []
        sub_dirs = []
        marker = ""
        count = 1
        while True:
            response = self.client.list_objects(self.bucket_name, self.prefix, delimiter, marker)
            count += 1
            if "CommonPrefixes" in response:
                common_prefixes = response.get("CommonPrefixes")
                sub_dirs.extend(common_prefixes)

            if "Contents" in response:
                contents = response.get("Contents")
                file_infos.extend(contents)

            if "NextMarker" in response.keys():
                marker = response["NextMarker"]
            else:
                break

        sorted(file_infos, key=lambda file_info: file_info["Key"])

        return file_infos

    def download_file_from_bucket(self, obj, local_file_path):
        response = self.client.download_file(
            Bucket=self.bucket_name,
            Key=obj,
            DestFilePath=local_file_path
        )

    def download_dataset(self, localDir, delimiter=''):
        try:
            file_infos = self.list_currentDir(delimiter)
        except CosServiceError as e:
            print(e.get_origin_msg())
            print(e.get_digest_msg())
            print(e.get_status_code())
            print(e.get_error_code())
            print(e.get_error_msg())
            print(e.get_resource_location())
            print(e.get_trace_id())
            print(e.get_request_id())

        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        for i in range(len(file_infos)):
            file_cos_key = file_infos[i]["Key"]
            cos_key = re.search(rf'{os.path.basename(self.prefix)}(.*)', file_infos[i]["Key"]).group(0)
            local_path= localDir + '/' + cos_key
            # 如果本地目录结构不存在，递归创建
            if not os.path.exists(os.path.dirname(local_path)):
                os.makedirs(os.path.dirname(local_path))
            pool.apply_async(func=self.download_file_from_bucket, args=(file_cos_key, local_path))
        pool.close()
        pool.join()
        return file_infos

    def up_file_to_bucket(self, up_bucket_path, local_path):
        response = self.client.upload_file(
            Bucket=self.bucket_name,
            Key=up_bucket_path,
            LocalFilePath=local_path,
            EnableMD5=False,
            progress_callback=None
        )

    def upload_dataset(self, local_file_path):
        g = os.walk(local_file_path)
        new_prefix = "/".join(self.prefix.split('/')[:-1])
        file_infos = []
        for path, dir_list, file_list in g:
            sorted(file_list, reverse=True)
            for file_name in file_list:
                local_file = os.path.join(path, file_name)
                result = re.search(rf'{os.path.basename(self.prefix)}(.*)', local_file)
                if result:
                    object_path = result.group(0)
                    cosObjectKey = str(new_prefix + '/' + object_path).replace('\\', '/')
                    file_infos.append({'local_file': local_file, 'cos_path': cosObjectKey})
        if file_infos:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            for i in range(len(file_infos)):
                local_path = file_infos[i]['local_file']
                up_bucket_path = file_infos[i]['cos_path']
                pool.apply_async(func=self.up_file_to_bucket, args=(up_bucket_path, local_path))
            pool.close()
            pool.join()
        else:
            # logger.error(f'{local_file_path} Not all files upload successed. you should retry')
            print(f"{local_file_path} Not all files upload successed. you should retry")
        return file_infos


def get_parma():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upload', action="store_true", default=False, help="will upload if present (default: download)")
    parser.add_argument('-c', '--config_dir', required=True, type=str, help='config file path')
    parser.add_argument('-t', '--cloud_type', required=True, type=str, help='config options: "local" for 公有云, "cloud" for 合规云')
    parser.add_argument('-b', '--bucket', required=True, type=str, help='bucket name')
    parser.add_argument('-p', '--prefix', required=True, type=str, help='cos prefix')
    parser.add_argument('-d', '--dir_path', required=True, type=str, help='local dir path')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parma()
    cld = cloudData(args.config_dir, args.cloud_type, args.bucket, args.prefix)
    if args.upload:
        l = cld.upload_dataset(args.dir_path)
        print(len(l))
    else:
        l = cld.download_dataset(args.dir_path)
        print(len(l))
