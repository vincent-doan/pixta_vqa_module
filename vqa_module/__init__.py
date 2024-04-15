import os
device_ids = [0, 1, 2, 3]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in device_ids)