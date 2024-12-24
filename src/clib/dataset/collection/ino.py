from typing import Callable, Optional
from pathlib import Path
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

__all__ = ['INO']

class INO(VisionDataset):
    file = {
        'INO Crossroads':{
            'url': 'https://inostorage.blob.core.windows.net/media/1546/ino_crossroads.zip',
            'md5': '0856f906e660290d3f581c9e71f5b544',
            'extract_root_modify': True,
            'addition': 'INO_Crossroads',
            'vis': True,
            'ir': False,
        },
        'INO Trees and runner':{
            'url': 'https://inostorage.blob.core.windows.net/media/2518/ino_treesandrunner.zip',
            'md5': 'd9fff5d4f4a982b8a2ea147c363fc62f',
            'extract_root_modify': False,
            'addition': 'INO_TreesAndRunner',
            'vis': True,
            'ir': True,
        },
        'INO Visitor parking':{
            'url': 'https://inostorage.blob.core.windows.net/media/2517/ino_visitorparking.zip',
            'md5': 'e1d9606f24ba421fc61194184b8800d2',
            'extract_root_modify': False,
            'addition': 'INO_VisitorParking',
            'vis': True,
            'ir': True,
        },
        'INO Main entrance':{
            'url': 'https://inostorage.blob.core.windows.net/media/2520/ino_mainentrance.zip',
            'md5': '9adb72f3f5daeeeb39167cff27776459',
            'extract_root_modify': False,
            'addition': 'INO_MainEntrance',
            'vis': True,
            'ir': False,
        },
        'INO Parking evening':{
            'url': 'https://inostorage.blob.core.windows.net/media/2519/ino_parkingevening.zip',
            'md5': 'a8010c369b174d7231734ba9e5a6dd46',
            'extract_root_modify': False,
            'addition': 'INO_ParkingEvening',
            'vis': True,
            'ir': True,
        },
        'INO Close person':{
            'url': 'https://inostorage.blob.core.windows.net/media/1551/ino_closeperson.zip',
            'md5': '99aac2b32d35db80000e78dcfb50c5bd',
            'extract_root_modify': False,
            'addition': 'INO_ClosePerson',
            'vis': True,
            'ir': True,
        },
        'INO Coat deposit':{
            'url': 'https://inostorage.blob.core.windows.net/media/1552/ino_coatdeposit.zip',
            'md5': '04f7c4bda7605639fbd321c4a44a411b',
            'extract_root_modify': True,
            'addition': 'INO_CoatDeposit',
            'vis': True,
            'ir': True,
        },
        'INO Multiple deposit':{
            'url': 'https://inostorage.blob.core.windows.net/media/1554/ino_multipledeposit.zip',
            'md5': '86e61100abcc96cfdf31e02d4a4eb418',
            'extract_root_modify': True,
            'addition': 'INO_MulitpleDeposit',
            'vis': True,
            'ir': True,
        },
        'INO Backyard runner':{
            'url': 'https://inostorage.blob.core.windows.net/media/1550/ino_backyardrunner.zip',
            'md5': 'f3611ed4bc5484807b3e3f9566f4c837',
            'extract_root_modify': True,
            'addition': 'INO_BackyardRunner',
            'vis': True,
            'ir': True,
        },
        'INO Group fight':{
            'url': 'https://inostorage.blob.core.windows.net/media/1553/ino_groupfight.zip',
            'md5': 'a9e289258592dd19d0ee7a80ddaea3fc',
            'extract_root_modify': True,
            'addition': 'INO_GroupFight',
            'vis': True,
            'ir': True,
        },
        'INO Parking snow':{
            'url': 'https://inostorage.blob.core.windows.net/media/1555/ino_parkingsnow.zip',
            'md5': '81e2df675174a299b030d1977869c442',
            'extract_root_modify': True,
            'addition': 'INO_ParkingSnow',
            'vis': True,
            'ir': True,
        },
        'Highway I':{
            'url': 'https://inostorage.blob.core.windows.net/media/1548/highwayi.zip',
            'md5': '99c35b03c278a6f3585307150f3d03cf',
            'extract_root_modify': True,
            'addition': 'HighwayI',
            'vis': False,
            'ir': False,
        },
        'Lobby':{
            'url': 'https://inostorage.blob.core.windows.net/media/1556/lobby.zip',
            'md5': '55b80978681510e16a764c7990c7132d',
            'extract_root_modify': True,
            'addition': 'Lobby',
            'vis': False,
            'ir': False,
        },
        'Campus':{
            'url': 'https://inostorage.blob.core.windows.net/media/1547/campus.zip',
            'md5': 'a43f27ef135a304da1b3806684688ef4',
            'extract_root_modify': True,
            'addition': 'Campus',
            'vis': False,
            'ir': False,
        },
        'Highway III':{
            'url': 'https://inostorage.blob.core.windows.net/media/1549/highwayiii.zip',
            'md5': '2eb33705a561847afebc022cba247f65',
            'extract_root_modify': True,
            'addition': 'HighwayIII',
            'vis': False,
            'ir': False,
        }
    }

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform)
        self._base_folder = Path(self.root) / "ino"

        if download:
            self.download()

    def _check_integrity(self,item):
        info = self.file[item]
        if not check_integrity(str(self._base_folder / f'{info["addition"].lower()}.zip'), info["md5"]):
            return False
            
        if not ((self._base_folder/item).exists() and (self._base_folder/item).is_dir()):
            return False

        return True
    
    def download(self):
        for item in self.file:
            if self._check_integrity(item):
                continue
            if self.file[item]['extract_root_modify']:
                extract_root = self._base_folder / self.file[item]['addition']
            else:
                extract_root = None
            download_and_extract_archive(
                url=f"{self.file[item]['url']}",
                download_root=self._base_folder,
                extract_root=extract_root,
                md5=self.file[item]['md5'],
            )

    
