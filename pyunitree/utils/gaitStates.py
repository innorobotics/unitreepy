import enum

class LegState(enum.Enum):
  """The state of a leg during locomotion."""
  SWING = 0
  STANCE = 1
  # A swing leg that collides with the ground.
  EARLY_CONTACT = 2
  # A stance leg that loses contact.
  LOSE_CONTACT = 3