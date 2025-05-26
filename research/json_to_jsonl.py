{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9631375c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import sys\n",
    "\n",
    "def convert_json_to_jsonl(json_file, jsonl_file):\n",
    "    # Đọc file JSON\n",
    "    with open(json_file, 'r', encoding='utf-8') as f:\n",
    "        data = json.load(f)  # data là một list các dict\n",
    "\n",
    "    # Ghi từng dòng sang JSONL\n",
    "    with open(jsonl_file, 'w', encoding='utf-8') as f:\n",
    "        for item in data:\n",
    "            f.write(json.dumps(item, ensure_ascii=False) + '\\n')\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    if len(sys.argv) != 3:\n",
    "        print(\"Usage: python convert_json_to_jsonl.py input.json output.jsonl\")\n",
    "    else:\n",
    "        convert_json_to_jsonl(sys.argv[1], sys.argv[2])\n",
    "        print(\"Done! Converted\", sys.argv[1], \"to\", sys.argv[2])\n"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
