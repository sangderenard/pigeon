from typing import Optional
import logging
import json


class JobGraphNode:
    def __init__(self, job_obj, node_id, name: Optional[str] = None):
        self.job = job_obj
        self.id = node_id
        if name is None:
            self.name = f"JobNode-{node_id}"
        else:
            self.name = name
        self.sinks = []
        self.sources = []
        self.tape = []      # <-- Operation history, in order
        self.trace = trace
    @property
    def children(self):
        return self.sinks
    @property
    def parents(self):
        return self.sources

class JobGraph:
    def __init__(self, trace=False):
        self.nodes = {}  # id -> JobGraphNode
        self.edges = []  # (from_id, to_id)
        self.trace = trace
        self.logger = logging.getLogger("JobGraph")
        if self.trace:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

    def _trace(self, msg):
        if self.trace:
            self.logger.debug(msg)

    def _record(self, action, **kwargs):
        # Append an event to the tape
        entry = {'action': action, **kwargs}
        self.tape.append(entry)
        if self.trace:
            self._record(f"TAPE: {entry}")

    def add_node(self, node: JobGraphNode):
        self.nodes[node.id] = node
        self._record(
            'add_node',
            node_id=node.id,
            name=node.name,
            job_type=type(node.job).__name__,
            job_repr=repr(node.job),
        )

    def add_edge(self, from_id, to_id):
        self.nodes[from_id].sinks.append(to_id)
        self.edges.append((from_id, to_id))
        self._record('add_edge', from_id=from_id, to_id=to_id)

    def get_downstreams(self, node_id):
        ds = [self.nodes[nid] for nid in self.nodes[node_id].sinks]
        self._record(f"GET_DOWNSTREAMS: node={node_id} -> {[n.id for n in ds]}")
        return ds
    
    def serialize_tape(self, path_or_file=None):
        data = json.dumps(self.tape, indent=2)
        if path_or_file:
            with open(path_or_file, 'w') as f:
                f.write(data)
        return data

    def load_tape(self, tape_json, job_registry=None):
        tape = json.loads(tape_json)
        for entry in tape:
            action = entry['action']
            if action == 'add_node':
                job_type = entry.get('job_type')
                # Use registry or fallback
                if job_registry and job_type in job_registry:
                    job_obj = job_registry[job_type]()
                else:
                    job_obj = None  # Or a stub/factory/default
                node = JobGraphNode(
                    job_obj=job_obj,
                    node_id=entry['node_id'],
                    name=entry.get('name'),
                )
                self.nodes[node.id] = node
            elif action == 'add_edge':
                from_id = entry['from_id']
                to_id = entry['to_id']
                self.nodes[from_id].sinks.append(to_id)
                self.edges.append((from_id, to_id))

