use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::Bound;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

thread_local!(
    static CALL_STACK: RefCell<Vec<GNodeKey>> = RefCell::new(Vec::new());
);

#[derive(Clone, Debug)]
struct GNodeKey {
    func_name: String,
    args: Py<PyTuple>,
    kwargs: PyObject, // Expecting frozendict
}

impl Hash for GNodeKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.func_name.hash(state);
        Python::with_gil(|py| {
            let args_hash = self.args.bind(py).hash().unwrap_or(-1);
            let kwargs_hash = self.kwargs.bind(py).hash().unwrap_or(-1);
            args_hash.hash(state);
            kwargs_hash.hash(state);
        });
    }
}

impl PartialEq for GNodeKey {
    fn eq(&self, other: &Self) -> bool {
        self.func_name == other.func_name &&
        Python::with_gil(|py| {
            let self_args = self.args.bind(py);
            let other_args = other.args.bind(py);
            let self_kwargs = self.kwargs.bind(py);
            let other_kwargs = other.kwargs.bind(py);
            self_args.eq(other_args).unwrap_or(false) &&
            self_kwargs.eq(other_kwargs).unwrap_or(false)
        })
    }
}

impl Eq for GNodeKey {}


#[derive(Clone)]
struct GNode {
    value: PyObject,
    children: Vec<GNodeKey>,
}

struct GraphState {
    nodes: HashMap<GNodeKey, GNode>,
}

impl GraphState {
    fn new() -> Self {
        GraphState { nodes: HashMap::new() }
    }

    fn get_ancestors(&self, key: &GNodeKey) -> HashSet<GNodeKey> {
        let mut ancestors = HashSet::new();
        let mut queue: VecDeque<GNodeKey> = self.nodes
            .iter()
            .filter(|(_, node)| node.children.contains(key))
            .map(|(k, _)| k.clone())
            .collect();

        for parent in &queue {
            ancestors.insert(parent.clone());
        }

        while let Some(parent_key) = queue.pop_front() {
            for (grandparent_key, grandparent_node) in &self.nodes {
                if grandparent_node.children.contains(&parent_key) {
                    if ancestors.insert(grandparent_key.clone()) {
                        queue.push_back(grandparent_key.clone());
                    }
                }
            }
        }
        ancestors
    }
}

#[pyclass(name = "Graph")]
struct Graph {
    state: Mutex<GraphState>,
}

#[pymethods]
impl Graph {
    #[new]
    fn new() -> Self {
        Graph { state: Mutex::new(GraphState::new()) }
    }

    fn execute(
        &self,
        py: Python,
        func: PyObject,
        func_name: String,
        args: &Bound<PyTuple>,
        kwargs: Option<&Bound<PyDict>>,
    ) -> PyResult<PyObject> {
        let frozendict_module = py.import_bound("frozendict")?;
        let frozendict_class = frozendict_module.getattr("frozendict")?;
        let frozen_kwargs = if let Some(kwargs_dict) = kwargs {
            frozendict_class.call1((kwargs_dict,))?.to_object(py)
        } else {
            frozendict_class.call0()?.to_object(py)
        };

        let key = GNodeKey {
            func_name,
            args: args.into_py(py),
            kwargs: frozen_kwargs,
        };

        // Record dependency on parent
        CALL_STACK.with(|stack| {
            if let Some(parent_key) = stack.borrow().last() {
                let mut state = self.state.lock().unwrap();
                let parent_node = state.nodes.entry(parent_key.clone()).or_insert_with(|| GNode {
                    value: py.None(),
                    children: Vec::new(),
                });
                if !parent_node.children.contains(&key) {
                    parent_node.children.push(key.clone());
                }
            }
        });

        // Check cache
        {
            let state = self.state.lock().unwrap();
            if let Some(node) = state.nodes.get(&key) {
                // Return value if it's not a placeholder
                if !node.value.is_none(py) {
                    return Ok(node.value.clone());
                }
            }
        }

        // Execute function
        CALL_STACK.with(|stack| stack.borrow_mut().push(key.clone()));
        let value = func.call_bound(py, args, kwargs)?;
        CALL_STACK.with(|stack| { stack.borrow_mut().pop(); });

        // Update node with value
        {
            let mut state = self.state.lock().unwrap();
            let node = state.nodes.entry(key.clone()).or_insert_with(|| GNode {
                value: py.None(),
                children: Vec::new(),
            });
            node.value = value.clone();
        }

        Ok(value)
    }

    fn override_value(
        &self,
        py: Python,
        func_name: String,
        value: PyObject,
        args: &Bound<PyTuple>,
        kwargs: Option<&Bound<PyDict>>,
    ) -> PyResult<()> {
        let frozendict_module = py.import_bound("frozendict")?;
        let frozendict_class = frozendict_module.getattr("frozendict")?;
        let frozen_kwargs = if let Some(kwargs_dict) = kwargs {
            frozendict_class.call1((kwargs_dict,))?.to_object(py)
        } else {
            frozendict_class.call0()?.to_object(py)
        };

        let key = GNodeKey {
            func_name,
            args: args.into_py(py),
            kwargs: frozen_kwargs,
        };

        let mut state = self.state.lock().unwrap();
        let ancestors = state.get_ancestors(&key);
        for a in ancestors {
            state.nodes.remove(&a);
        }

        let node = GNode {
            value,
            children: vec![],
        };
        state.nodes.insert(key, node);

        Ok(())
    }

    fn clear(&self) {
        self.state.lock().unwrap().nodes.clear();
    }

    fn get_edges(&self, py: Python) -> PyResult<Vec<(String, String)>> {
        let state = self.state.lock().unwrap();
        let mut edges = Vec::new();

        for (parent_key, parent_node) in &state.nodes {
            for child_key in &parent_node.children {
                let parent_repr = format!(
                    "{}(*{}, **{})",
                    parent_key.func_name,
                    parent_key.args.bind(py).repr()?,
                    parent_key.kwargs.bind(py).repr()?
                );
                let child_repr = format!(
                    "{}(*{}, **{})",
                    child_key.func_name,
                    child_key.args.bind(py).repr()?,
                    child_key.kwargs.bind(py).repr()?
                );
                edges.push((parent_repr, child_repr));
            }
        }
        Ok(edges)
    }
}

#[pymodule]
fn kinyu_graph(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Graph>()?;
    Ok(())
}