tidy_folder = '../raw-data/tidy-data/'
# tidy_folder = '../raw-data/independent-validation/'
EXCEL_PATH = '../raw-data/20201027.xlsx'
RESULT_RECORD_FILE = '../record/result.csv'
# excel_path = '../raw-data/验证集表格.xlsx'
model_dir = '../models/video/'
data_dir = '../data/video/'
record_dir = '../record/video/'
result_dir = '../result/video/'
writer_dir = '../tensorboard/video/'
crop_dir = '../data/video/auto-selection'
roi_dir = '../data/video/20200619-segmented'
allocate = '../raw-data/20200625-allocate.txt'
e_seg_model_path = '../models/segmentation/20200617-e.pth'
excel_keys = {'序号': 'id', '医院': 'hospital', '日期': 'date', '姓名': 'name', '性别': 'sex', '年龄': 'age',
              '位置': 'local', '诊断': 'diag', '小类': 'fine', '大类': 'coarse', '良恶性': 'BM'}

HOSPITAL_KEY = {'上海胸科': 'shanghai', '河南省医': 'henan', '烟台硫磺顶': 'yantai', '杭州市一': 'hangzhou',
                '安徽胸科': 'anhui'}

IMAGE_FORMAT = '.png'

B_IMAGE_SIZE = 224
F_IMAGE_SIZE = 224
E_DIMS = 512

MODE_SIGN = {'doppler': 'F', 'elastic': 'E', 'gray': 'B', 'CT': 'C'}
BASIC_FOLDER = {'F': 'raw-img', 'E': 'elastic', 'B': 'raw-img', 'C': 'e-raw-volume'}

CROP_RANGE = {
    # 'shanghai': {'1920': {'B': (1238, 1807, 300, 769), 'E': (1466, 1807, 300, 690, 368), 'F': (1238, 1807, 300, 769)},
    #              '1024': {'B': (79, 880, 143, 616), 'E': (88, 520, 56, 512)}},

    'shanghai': {'1920': {'B': (1238, 1807, 300, 769), 'E': (1396, 1791, 267, 715, 425), 'F': (1238, 1807, 300, 769)},
                 '1024': {'B': (79, 880, 143, 616), 'E': (88, 520, 56, 512)}},

    'henan': {'1920': {'B': (970, 1540, 136, 560), 'E': (960, 1540, 137, 799, 624), 'F': (577, 1540, 136, 976)}},

    'anhui': {'1280': {'B': (368, 1220, 77, 908), 'E': (639, 1220, 86, 738, 624), 'F': (368, 1220, 86, 908)}},

    'hangzhou': {'1920': {'B': (575, 1540, 155, 970), 'E': (960, 1540, 144, 798, 624), 'F': (576, 1539, 136, 970)}},

    'yantai': {'1280': {'B': (256, 1220, 77, 910), 'E': (640, 1220, 77, 737, 623), 'F': (256, 1220, 77, 910)}},
    }
