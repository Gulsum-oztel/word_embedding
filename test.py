from datasets import load_dataset

# Veri setini yükle
ds = load_dataset("We-Bears/Turkish-Review-Sentiment-Data")

# Veri hakkında genel bilgi al
print(ds)

# İlk 3 örneği gör
print(ds['train'][:3])
