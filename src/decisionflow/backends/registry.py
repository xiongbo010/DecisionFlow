"""Small dependency-free registries for pluggable inference backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Tuple


BackendFactory = Callable[..., Any]


@dataclass(frozen=True)
class BackendDescriptor:
    name: str
    factory: BackendFactory
    capabilities: Tuple[str, ...]
    exact: bool
    description: str = ""
    aliases: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Return the provider-neutral metadata exposed by tools and the CLI."""
        return {
            "name": self.name,
            "capabilities": list(self.capabilities),
            "exact": self.exact,
            "description": self.description,
            "aliases": list(self.aliases),
        }


class BackendRegistry:
    """Map stable backend names to lazy factories and capability metadata."""

    def __init__(self) -> None:
        self._descriptors: Dict[str, BackendDescriptor] = {}
        self._aliases: Dict[str, str] = {}

    def register(
        self,
        name: str,
        factory: BackendFactory,
        *,
        capabilities: Iterable[str],
        exact: bool,
        description: str = "",
        aliases: Iterable[str] = (),
        replace: bool = False,
    ) -> None:
        canonical = str(name).strip().lower()
        if not canonical:
            raise ValueError("backend name cannot be empty")
        alias_tuple = tuple(str(alias).strip().lower() for alias in aliases)
        occupied = {canonical, *alias_tuple}
        if not replace:
            conflicts = sorted(
                item
                for item in occupied
                if item in self._descriptors or item in self._aliases
            )
            if conflicts:
                raise ValueError("backend name already registered: %s" % conflicts[0])
        if replace:
            for item in occupied:
                old = self._aliases.pop(item, None)
                if old is not None:
                    descriptor = self._descriptors.get(old)
                    if descriptor is not None:
                        self._descriptors[old] = BackendDescriptor(
                            name=descriptor.name,
                            factory=descriptor.factory,
                            capabilities=descriptor.capabilities,
                            exact=descriptor.exact,
                            description=descriptor.description,
                            aliases=tuple(
                                alias for alias in descriptor.aliases if alias != item
                            ),
                        )
                self._descriptors.pop(item, None)
        descriptor = BackendDescriptor(
            name=canonical,
            factory=factory,
            capabilities=tuple(dict.fromkeys(map(str, capabilities))),
            exact=bool(exact),
            description=str(description),
            aliases=alias_tuple,
        )
        self._descriptors[canonical] = descriptor
        for alias in alias_tuple:
            self._aliases[alias] = canonical

    def descriptor(self, name: str) -> BackendDescriptor:
        key = str(name).strip().lower()
        canonical = self._aliases.get(key, key)
        try:
            return self._descriptors[canonical]
        except KeyError as error:
            raise ValueError("unknown backend: %s" % name) from error

    def create(self, name: str, **options: Any) -> Any:
        return self.descriptor(name).factory(**options)

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._descriptors))

    def describe(self) -> Mapping[str, BackendDescriptor]:
        return {name: self._descriptors[name] for name in self.names()}

    def supports(self, name: str, queries: Iterable[str]) -> bool:
        capabilities = set(self.descriptor(name).capabilities)
        return set(queries).issubset(capabilities)
