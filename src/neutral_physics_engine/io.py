"""
io.py

This module provides high-performance data logging for N-body gravitational
simulations using the Hierarchical Data Format (HDF5). It is designed to handle
the massive data throughput required by large-scale physical simulations, ensuring
efficient disk I/O without bottlenecking the main integration loop.

The HDF5Writer class implements a buffered writing strategy, caching simulation
states (positions, velocities, energies, and momenta) in memory and writing them
to disk in large, chunked blocks. It uses extendable datasets to allow simulations
of arbitrary or undetermined length.

Key features:
- In-memory buffering to minimize expensive disk I/O operations
- ZLIB/GZIP compression with adjustable effort for storage optimization
- Extendable chunked HDF5 datasets (`maxshape=(None, ...)`)
- Automatic metadata tagging (timestamps, format versioning, system parameters)
- Context manager support for safe resource handling and crash recovery

Dependencies:
    h5py  : Pythonic interface to the HDF5 binary data format
    numpy : Used for pre-allocating contiguous memory buffers
"""

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import h5py


class HDF5Writer:
    """
    Manages buffered, chunked data output to an HDF5 file for N-body simulations.

    This class sets up the necessary HDF5 groups and datasets to track the time
    evolution of kinematic variables (positions, velocities) and global system
    diagnostics (kinetic/potential energy, linear/angular momentum).

    It uses an internal NumPy buffer to store step data. Once the buffer is full,
    the underlying HDF5 datasets are resized and the buffer is flushed to disk.
    """

    def __init__(
        self,
        filename: str | Path,
        n_bodies: int,
        buffer_size: int,
        compression: str = "gzip",
        compression_opts: int = 4,
        metadata: dict | None = None,
    ):
        """
        Initialize the HDF5 writer and allocate internal memory buffers.

        Parameters:
        -----------
        filename : str | Path
            The destination path for the HDF5 output file.
        n_bodies : int
            The number of bodies in the simulation (determines dataset shapes).
        buffer_size : int
            The number of simulation steps to hold in memory before flushing to disk.
            Larger buffers reduce I/O overhead but increase RAM usage.
        compression : str, optional
            The compression filter to use for HDF5 datasets (default "gzip").
        compression_opts : int, optional
            The compression level/effort, usually from 0 to 9 (default 4).
        metadata : dict | None, optional
            Additional key-value pairs to store in the HDF5 root metadata attributes.
        """
        self.filename = Path(filename)
        self.n_bodies = n_bodies
        self.buffer_size = buffer_size
        self.compression = compression
        self.compression_opts = compression_opts
        self.metadata = metadata or {}
        self.file = None
        self.buffer_idx = 0
        self.write_idx = 0
        self._buffer = {
            "time": np.empty(self.buffer_size, dtype=np.float64),
            "positions": np.empty(
                (self.buffer_size, self.n_bodies, 3), dtype=np.float64
            ),
            "velocities": np.empty(
                (self.buffer_size, self.n_bodies, 3), dtype=np.float64
            ),
            "kinetic": np.empty(self.buffer_size, dtype=np.float64),
            "potential": np.empty(self.buffer_size, dtype=np.float64),
            "total": np.empty(self.buffer_size, dtype=np.float64),
            "linear_p": np.empty(self.buffer_size, dtype=np.float64),
            "angular_p": np.empty(self.buffer_size, dtype=np.float64),
        }

    def _setup_file(self):
        """
        Create the HDF5 file, write metadata, and initialize extendable datasets.

        This method is called automatically when entering the context manager.
        It configures the chunk sizes to match the buffer size, optimizing the
        alignment of disk writes.

        Raises:
        -------
        FileExistsError
            If the specified target filename already exists.
        """
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        if self.filename.exists():
            FileExistsError(f"{self.filename} already exists")
        self.file = h5py.File(self.filename, "w")

        # Metadata
        meta = self.file.create_group("metadata")
        timestamp = datetime.now(timezone.utc).isoformat()
        meta.attrs.update(
            {
                "n_bodies": self.n_bodies,
                "format": "neutral_physics_engine",
                "buffer_size": self.buffer_size,
                "compression": self.compression,
                "compression_opts": self.compression_opts,
                "created_on": timestamp,
                **self.metadata,
            }
        )

        # Extendable Dataset
        self.file.create_dataset(
            "time",
            shape=(0,),
            maxshape=(None,),
            dtype="f8",
            chunks=(self.buffer_size,),
            compression=self.compression,
            compression_opts=self.compression_opts,
        )

        self.file.create_dataset(
            "positions",
            shape=(0, self.n_bodies, 3),
            maxshape=(None, self.n_bodies, 3),
            dtype="f8",
            chunks=(self.buffer_size, self.n_bodies, 3),
            compression=self.compression,
            compression_opts=self.compression_opts,
        )

        self.file.create_dataset(
            "velocities",
            shape=(0, self.n_bodies, 3),
            maxshape=(None, self.n_bodies, 3),
            dtype="f8",
            chunks=(self.buffer_size, self.n_bodies, 3),
            compression=self.compression,
            compression_opts=self.compression_opts,
        )

        for name in ("kinetic", "potential", "total", "linear_p", "angular_p"):
            self.file.create_dataset(
                (
                    f"energy/{name}"
                    if name in ("kinetic", "potential", "total")
                    else f"momentum/{name}"
                ),
                shape=(0,),
                maxshape=(None,),
                dtype="f8",
                chunks=(self.buffer_size,),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

    def append_step(
        self,
        t: float,
        positions: np.ndarray,
        velocities: np.ndarray,
        kinetic: float,
        potential: float,
        total: float,
        linear_p: float,
        angular_p: float,
    ) -> None:
        """
        Append a single simulation time step to the internal memory buffer.

        If appending this step fills the buffer, the buffer is automatically
        flushed to the HDF5 file on disk.

        Parameters:
        -----------
        t : float
            Current simulation time.
        positions : np.ndarray
            Array of body positions (shape: (n_bodies, 3)).
        velocities : np.ndarray
            Array of body velocities (shape: (n_bodies, 3)).
        kinetic : float
            Total kinetic energy of the system.
        potential : float
            Total gravitational potential energy of the system.
        total : float
            Total mechanical energy of the system.
        linear_p : float
            Magnitude of the total linear momentum.
        angular_p : float
            Magnitude of the total angular momentum.
        """
        i = self.buffer_idx
        self._buffer["time"][i] = t
        self._buffer["positions"][i] = positions
        self._buffer["velocities"][i] = velocities
        self._buffer["kinetic"][i] = kinetic
        self._buffer["potential"][i] = potential
        self._buffer["total"][i] = total
        self._buffer["linear_p"][i] = linear_p
        self._buffer["angular_p"][i] = angular_p

        self.buffer_idx += 1

        if self.buffer_idx >= self.buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """
        Resize the HDF5 datasets and write the current buffer contents to disk.

        This method extends the time-axis of all datasets by the current number
        of buffered steps, dumps the NumPy arrays into the HDF5 file, and resets
        the buffer index to zero.
        """
        if self.buffer_idx == 0 or self.file is None:
            return

        new_size = self.write_idx + self.buffer_idx

        # Resize Dataset
        self.file["time"].resize(new_size, axis=0)
        self.file["positions"].resize((new_size, self.n_bodies, 3))
        self.file["velocities"].resize((new_size, self.n_bodies, 3))

        for name in ("kinetic", "potential", "total"):
            self.file[f"energy/{name}"].resize(new_size, axis=0)

        for name in ("linear_p", "angular_p"):
            self.file[f"momentum/{name}"].resize(new_size, axis=0)

        # Write Buffer into Dataset
        idx = slice(self.write_idx, self.write_idx + self.buffer_idx)
        self.file["time"][idx] = self._buffer["time"][: self.buffer_idx]
        self.file["positions"][idx, ...] = self._buffer["positions"][: self.buffer_idx]
        self.file["velocities"][idx, ...] = self._buffer["velocities"][
            : self.buffer_idx
        ]

        for name in ("kinetic", "potential", "total"):
            self.file[f"energy/{name}"][idx] = self._buffer[name][: self.buffer_idx]

        for name in ("linear_p", "angular_p"):
            self.file[f"momentum/{name}"][idx] = self._buffer[name][: self.buffer_idx]

        # Clear Buffer
        self.buffer_idx = 0
        self.write_idx = new_size

        if self.write_idx % 1000 == 0:
            self.file.flush()

    def flush(self) -> None:
        """
        Public interface to force a buffer flush to disk.
        Useful for ensuring data integrity before long computational pauses.
        """
        if self.buffer_idx > 0:
            self._flush_buffer()

    def close(self) -> None:
        """
        Flush any remaining buffered data and safely close the HDF5 file handle.
        """
        self.flush()
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        """
        Enter the runtime context related to this object.
        Initializes the HDF5 file and datasets.
        """
        self._setup_file()
        return self

    def __exit__(self, exc_type, exc, tb):
        """
        Exit the runtime context.
        Ensures the file is safely closed and buffered data is flushed, even
        if the simulation encounters an exception.
        """
        if exc_type is not None:
            print(f"Simulation Crashed: {exc}")
        self.close()
        return False
