from data.dataset import Text2SpeechDataset, BucketSampler
from torch.utils.data import DataLoader
from data.utils import make_filelist


tts_dataset = Text2SpeechDataset(file_path='filelist.txt',
                                 sample_rate=16000)
bucket_sampler = BucketSampler(dataset=tts_dataset)
dataloader = DataLoader(dataset=tts_dataset, batch_size=1,
                        batch_sampler=bucket_sampler, collate_fn=bucket_sampler.collate_fn)

for idx, data in enumerate(dataloader):
    print(f'Spec: {data[0].size()}')
    print(f'Audio: {data[1].size()}')
    print(f'Phoneme: {data[2].size()}')
    print(f'Duration: {data[3].size()}')
# make_filelist(data_dir='infore')