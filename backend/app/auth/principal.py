from __future__ import annotations

import re
from dataclasses import dataclass, field

_LABEL = re.compile(r"^[a-z][a-z0-9_-]*:[^\s]+$")

PERMISSION_CHAT = "chat"
PERMISSION_CONVERSATION_READ = "conversation:read"
PERMISSION_CONVERSATION_WRITE = "conversation:write"
PERMISSION_ADMIN = "admin"


class ZeroLabelPrincipalError(ValueError):
    """A principal without access labels is refused, never treated as public."""


@dataclass(frozen=True)
class Principal:
    """The authenticated caller for one request.

    ``labels`` are namespaced (``user:<sub>``, ``role:<role>``, ``group:<group>``,
    ``tenant:<id>``) and are compiled into search SQL as access predicates.
    """

    tenant_id: str
    principal_id: str
    labels: frozenset[str]
    permissions: frozenset[str] = field(default_factory=frozenset)
    username: str | None = None

    def __post_init__(self) -> None:
        if not self.labels:
            raise ZeroLabelPrincipalError("principal has no access labels")
        for label in self.labels:
            if not _LABEL.match(label):
                raise ValueError(f"label '{label}' is not namespaced (expected 'namespace:value')")

    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions
