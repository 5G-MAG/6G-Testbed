import sys
sys.path.insert(0, 'aitestbed/server')
import patch_host_sglang as patcher

calls = [0]
real = patcher.main
def counting(*a, **kw):
    calls[0] += 1
    return real(*a, **kw)
patcher.main = counting

# __init__ path (the production entry): constructing the client ensures
sys.path.insert(0, '.')
from clients.encoder_placement_client import EncoderPlacementClient
import hashlib, os, sglang
qv = os.path.join(os.path.dirname(sglang.__file__), 'srt', 'multimodal', 'processors', 'qwen_vl.py')
h0 = hashlib.sha256(open(qv, 'rb').read()).hexdigest()
c1 = EncoderPlacementClient()          # __init__ -> ensure (quiet, no-op)
c2 = EncoderPlacementClient()          # same process -> flag hit, no call
h1 = hashlib.sha256(open(qv, 'rb').read()).hexdigest()

from encpl_video_sampling import verify_frame_ratio_patch
verify_frame_ratio_patch(active=True)

print('patcher_calls (expect 1):', calls[0])
print('file_unchanged (expect True):', h0 == h1)
print('canary: OK (v3 semantics active)')
