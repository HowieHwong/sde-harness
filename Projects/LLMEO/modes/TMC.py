from pydantic import BaseModel

class TMC(BaseModel):
    center_metal: str
    ligands: list[str]
    expected_homolumo_gap: float

class ten_tmc(BaseModel):
    tmc: list[TMC]
    gap: list[float]

