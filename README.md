# vqa-module-for-image-search

![Experimental results](/assets/test_result.png "Experimental results")

```
git clone https://github.com/vincent-doan/pixta_vqa_module.git
```

1. Create a folder ``imgs`` at the root of the repository.
2. Add images to process
3. Modify questions, question weights, expected answers, and threshold in ``query_details.json``

```
./run_service.sh
```

```
python3 TEST_CLIENT.py --host ip_addr --port port_num --batch_size 100 --total_images 200 --query_details ./query_details.json
```