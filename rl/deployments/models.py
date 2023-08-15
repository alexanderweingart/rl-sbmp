from pydantic import BaseModel


class ActionAckermannFirstOrder(BaseModel):
    lin_vel: float
    phi: float
