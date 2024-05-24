import requests # pip install requests
import struct

def parse_single_file(url):
    # Fetch the first 8 bytes of the file
    headers = {'Range': 'bytes=0-7'}
    response = requests.get(url, headers=headers)
    # Interpret the bytes as a little-endian unsigned 64-bit integer
    length_of_header = struct.unpack('<Q', response.content)[0]
    # Fetch length_of_header bytes starting from the 9th byte
    headers = {'Range': f'bytes=8-{7 + length_of_header}'}
    response = requests.get(url, headers=headers)
    # Interpret the response as a JSON object
    header = response.json()
    return header

vector_file = 'output/vectors.safetensors'
header = parse_single_file(vector_file)


print(header)