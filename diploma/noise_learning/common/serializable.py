class DictSerializable:
    @classmethod
    def from_dict(cls, data: dict) -> 'DictSerializable':
        return cls(**data)

    def to_dict(self) -> dict:
        return vars(self)