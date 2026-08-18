use crate::{
    Batcher, DataLoader, Dataset,
    common::VecDataset,
    transform::{Map, MapDataset},
    utils,
};
use flate2::bufread::GzDecoder;
use luma_tensor::{
    Device,
    tensor::{FloatTensor, IntTensor},
};
use std::{
    convert::Infallible,
    fs::File,
    io::{Read, Seek, SeekFrom},
    marker::PhantomData,
    path::{Path, PathBuf},
};

// CVDF mirror of http://yann.lecun.com/exdb/mnist/
const URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist/";
const TRAIN_IMAGES: &str = "train-images-idx3-ubyte";
const TRAIN_LABELS: &str = "train-labels-idx1-ubyte";
const TEST_IMAGES: &str = "t10k-images-idx3-ubyte";
const TEST_LABELS: &str = "t10k-labels-idx1-ubyte";

const WIDTH: usize = 28;
const HEIGHT: usize = 28;

#[derive(Debug, Clone)]
pub struct MnistItem {
    pub image: [[f32; WIDTH]; HEIGHT],
    pub label: u8,
}

#[derive(Debug, Clone)]
struct MnistItemRaw {
    pub image_bytes: Vec<u8>,
    pub label: u8,
}

struct BytesToImage;

impl Map for BytesToImage {
    type Item = MnistItemRaw;
    type Output = MnistItem;

    fn map(&self, item: MnistItemRaw) -> MnistItem {
        assert_eq!(item.image_bytes.len(), WIDTH * HEIGHT);

        let mut image_array = [[0f32; WIDTH]; HEIGHT];
        for (i, pixel) in item.image_bytes.iter().enumerate() {
            let x = i % WIDTH;
            let y = i / WIDTH;
            image_array[y][x] = *pixel as f32;
        }

        MnistItem { image: image_array, label: item.label }
    }
}

type MnistDatasetImpl = MapDataset<VecDataset<MnistItemRaw>, BytesToImage>;

pub struct MnistDataset {
    dataset: MnistDatasetImpl,
}

impl Dataset for MnistDataset {
    type Item = MnistItem;
    type Error = Infallible;

    fn get(&self, index: usize) -> Result<Option<MnistItem>, Self::Error> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl MnistDataset {
    pub fn train<P: AsRef<Path>>(cache_dir: Option<P>) -> MnistResult<Self> {
        Self::new(MnistSplit::Train, cache_dir)
    }

    pub fn test<P: AsRef<Path>>(cache_dir: Option<P>) -> MnistResult<Self> {
        Self::new(MnistSplit::Test, cache_dir)
    }

    fn new<P: AsRef<Path>>(split: MnistSplit, cache_dir: Option<P>) -> MnistResult<Self> {
        let (image_path, label_path) = Self::download(split, cache_dir)?;
        let images = Self::read_images(&image_path)?;
        let labels = Self::read_labels(&label_path)?;

        let items: Vec<_> = images.into_iter().zip(labels).map(|(image_bytes, label)| MnistItemRaw { image_bytes, label }).collect();

        let dataset = VecDataset::new(items);
        let dataset = MapDataset::new(dataset, BytesToImage);

        Ok(Self { dataset })
    }

    fn read_images<P: AsRef<Path>>(file_path: &P) -> MnistResult<Vec<Vec<u8>>> {
        // Read number of images from 16-byte header metadata
        let mut f = File::open(file_path)?;
        let mut buf = [0u8; 4];
        f.seek(SeekFrom::Start(4))?;
        f.read_exact(&mut buf)?;
        let size = u32::from_be_bytes(buf);

        let mut buf_image: Vec<u8> = vec![0u8; WIDTH * HEIGHT * (size as usize)];
        f.seek(SeekFrom::Start(16))?;
        f.read_exact(&mut buf_image)?;

        let images = buf_image.chunks(WIDTH * HEIGHT).map(|chunk| chunk.to_vec()).collect();

        Ok(images)
    }

    fn read_labels<P: AsRef<Path>>(file_path: &P) -> MnistResult<Vec<u8>> {
        // Read number of labels from 8-byte header metadata
        let mut f = File::open(file_path)?;
        let mut buf = [0u8; 4];
        f.seek(SeekFrom::Start(4))?;
        f.read_exact(&mut buf)?;
        let size = u32::from_be_bytes(buf);

        let mut buf_labels: Vec<u8> = vec![0u8; size as usize];
        f.seek(SeekFrom::Start(8))?;
        f.read_exact(&mut buf_labels)?;

        Ok(buf_labels)
    }

    fn download<P: AsRef<Path>>(split: MnistSplit, cache_dir: Option<P>) -> MnistResult<(PathBuf, PathBuf)> {
        match cache_dir {
            Some(p) => Self::do_download(split, p.as_ref()),
            None => {
                let cache_dir = dirs::home_dir().expect("Could not get home directory").join(".cache").join("luma-dataset");
                Self::do_download(split, &cache_dir)
            }
        }
    }

    fn do_download(split: MnistSplit, cache_dir: &Path) -> MnistResult<(PathBuf, PathBuf)> {
        let split_dir = cache_dir.join("mnist").join(split.as_str());
        if !split_dir.exists() {
            std::fs::create_dir_all(&split_dir)?;
        }

        let (train_name, label_name) = match split {
            MnistSplit::Train => (TRAIN_IMAGES, TRAIN_LABELS),
            MnistSplit::Test => (TEST_IMAGES, TEST_LABELS),
        };

        Ok((Self::download_file(train_name, &split_dir)?, Self::download_file(label_name, &split_dir)?))
    }

    fn download_file<P: AsRef<Path>>(name: &str, dest_dir: &P) -> MnistResult<PathBuf> {
        let file_name = dest_dir.as_ref().join(name);

        if !file_name.exists() {
            // download gzip file
            let bytes = utils::download_file_as_bytes(&format!("{URL}{name}.gz"), name)?;
            // create file to write the downloaded content to
            let mut output_file = File::create(&file_name)?;
            // Decode gzip file content and write to disk
            let mut gz_buffer = GzDecoder::new(&bytes[..]);
            std::io::copy(&mut gz_buffer, &mut output_file)?;
        }

        Ok(file_name)
    }
}

#[derive(Clone)]
pub struct MnistBatch<D: Device> {
    pub images: FloatTensor<D>,
    pub targets: IntTensor<D>,
}

#[derive(Default)]
pub struct MnistBatcher<D: Device> {
    pub device: D,
    _data: PhantomData<D>,
}

impl<D: Device> MnistBatcher<D> {
    pub fn new(device: D) -> Self {
        Self { device, _data: Default::default() }
    }
}

impl<D: Device> Batcher for MnistBatcher<D> {
    type Item = MnistItem;
    type Output = MnistBatch<D>;
    type Error = MnistError;

    fn batch(&self, items: Vec<MnistItem>) -> MnistResult<MnistBatch<D>> {
        let batch_size = items.len();

        let mut flat_pixels = Vec::with_capacity(batch_size * HEIGHT * WIDTH);
        let mut labels = Vec::with_capacity(batch_size);

        for item in items {
            for row in item.image.iter() {
                flat_pixels.extend_from_slice(row);
            }
            labels.push(item.label as u32);
        }

        let images = FloatTensor::new(flat_pixels, &self.device)?.reshape((batch_size, HEIGHT, WIDTH))?; // [B, 28, 28]

        let images = ((images / 255.0) - 0.1307) / 0.3081;
        let targets = IntTensor::new(labels, &self.device)?.reshape((batch_size, 1))?;

        Ok(MnistBatch { images, targets })
    }
}

pub type MnistDataLoader<D> = DataLoader<MnistDataset, MnistBatcher<D>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MnistSplit {
    Train,
    Test,
}

impl MnistSplit {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Test => "test",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MnistError {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Utils(#[from] crate::utils::UtilError),

    #[error(transparent)]
    Core(#[from] luma_tensor::Error),
}

pub type MnistResult<T> = Result<T, MnistError>;
