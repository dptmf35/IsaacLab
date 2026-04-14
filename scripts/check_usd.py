"""Diagnostic script: inspect USD for physics issues.

Usage:
    ./isaaclab.sh -p scripts/check_usd.py
"""

import math
from pxr import Usd, UsdPhysics, UsdGeom, Gf

USD_PATH = "/home/yeseul/Desktop/Digital_Twin_UR/UR_Robot_System.usd"

print("=" * 60)
print(f"Inspecting: {USD_PATH}")
print("=" * 60)

stage = Usd.Stage.Open(USD_PATH)
if not stage:
    print("ERROR: Could not open USD file!")
    exit(1)

# 1. All FixedJoints
print("\n[1] FixedJoints:")
found_any = False
for prim in stage.Traverse():
    joint = UsdPhysics.FixedJoint(prim)
    if joint:
        found_any = True
        b0 = joint.GetBody0Rel().GetTargets()
        b1 = joint.GetBody1Rel().GetTargets()
        print(f"  {prim.GetPath()}")
        print(f"    body0={b0}  body1={b1}")
if not found_any:
    print("  (none)")

# 2. All Joints (any type)
print("\n[2] All Joints (any type):")
found_any = False
for prim in stage.Traverse():
    if UsdPhysics.Joint(prim):
        found_any = True
        j = UsdPhysics.Joint(prim)
        b0 = j.GetBody0Rel().GetTargets()
        b1 = j.GetBody1Rel().GetTargets()
        print(f"  {prim.GetPath()} [{prim.GetTypeName()}]")
        print(f"    body0={b0}  body1={b1}")
if not found_any:
    print("  (none)")

# 3. Prims with ArticulationRootAPI
print("\n[3] ArticulationRootAPI prims:")
for prim in stage.Traverse():
    if UsdPhysics.ArticulationRootAPI(prim):
        print(f"  {prim.GetPath()}")

# 4. Prims with RigidBodyAPI
print("\n[4] RigidBodyAPI prims:")
for prim in stage.Traverse():
    if UsdPhysics.RigidBodyAPI(prim):
        rb = UsdPhysics.RigidBodyAPI(prim)
        grav = rb.GetDisableGravityAttr().Get()
        print(f"  {prim.GetPath()}  disableGravity={grav}")

# 5. Check for degenerate scale (any component == 0)
print("\n[5] Prims with potentially degenerate scale:")
found_any = False
for prim in stage.Traverse():
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        continue
    ops = xformable.GetOrderedXformOps()
    for op in ops:
        if "scale" in op.GetOpName().lower():
            val = op.Get()
            if val is None:
                continue
            # val might be Vec3f/Vec3d
            try:
                components = [val[0], val[1], val[2]]
                if any(abs(c) < 1e-9 for c in components):
                    print(f"  ZERO SCALE: {prim.GetPath()}  scale={val}")
                    found_any = True
                elif any(math.isnan(c) or math.isinf(c) for c in components):
                    print(f"  NaN/Inf SCALE: {prim.GetPath()}  scale={val}")
                    found_any = True
            except Exception:
                pass
if not found_any:
    print("  (none detected)")

# 6. Sublayers
print("\n[6] Stage sublayers:")
root_layer = stage.GetRootLayer()
print(f"  Root: {root_layer.identifier}")
for sub in root_layer.subLayerPaths:
    print(f"  Sublayer: {sub}")

print("\n" + "=" * 60)
print("Done.")
