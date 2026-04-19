
const metadata = {};
const metrics = {};

metadata.Attachment = class {

    constructor() {
        this.metadata = new metadata.Attachment.Container('metadata');
        this.metrics = new metadata.Attachment.Container('metrics');
    }

    async open(context) {
        context = new view.Context(context);
        if (context.identifier.toLowerCase().endsWith('.json')) {
            const data = await context.peek('json');
            if (data && data.signature === 'netron:attachment') {
                const containers = [this.metadata, this.metrics];
                for (const container of containers) {
                    container.open(data[container.name]);
                }
                return true;
            }
        }
        return false;
    }

};

metadata.Attachment.Container = class {

    constructor(name) {
        this._name = name;
        this._entries = new Map();
    }

    get name() {
        return this._name;
    }

    open(data) {
        this._entries.clear();
        if (Array.isArray(data)) {
            for (const item of data) {
                if (item.kind && ('target' in item || 'identifier' in item)) {
                    const key = 'target' in item ? `${item.kind}::${item.target}` : `${item.kind}[${item.identifier}]`;
                    if (!this._entries.has(key)) {
                        this._entries.set(key, new Map());
                    }
                    const entries = this._entries.get(key);
                    entries.set(item.name, { value: item.value, type: item.type });
                }
            }
        }
    }

    model(value) {
        return this._list(value, 'model');
    }

    graph(value) {
        return this._list(value, 'graph');
    }

    node(value) {
        return this._list(value, 'node');
    }

    value(value) {
        return this._list(value, 'value');
    }

    tensor(value) {
        return this._list(value, 'tensor');
    }

    _list(value, kind) {
        const category = this._name;
        const entries = value[category] || [];
        const result = new Map(entries.map((entry) => [entry.name, entry]));
        if (value.name || kind === 'model' || kind === 'graph') {
            const key = `${kind}::${(value.name || '').split('\n').shift()}`;
            if (this._entries.has(key)) {
                for (const [name, entry] of this._entries.get(key)) {
                    const argument = new metadata.Argument(name, entry.value, entry.type || 'attribute');
                    result.set(name, argument);
                }
            }
        }
        if (value.identifier) {
            const key = `${kind}[${value.identifier}]`;
            if (this._entries.has(key)) {
                for (const [name, entry] of this._entries.get(key)) {
                    const argument = new metadata.Argument(name, entry.value, entry.type || 'attribute');
                    result.set(name, argument);
                }
            }
        }
        return Array.from(result.values());
    }
};

metadata.Argument = class {

    constructor(name, value, type) {
        this.name = name;
        this.value = value;
        this.type = type;
    }
};

metrics.Model = class {

    constructor(model) {
        this._model = model;
        this._metrics = null;
    }

    get metrics() {
        if (this._metrics === null) {
            this._metrics = [];
            this._metrics = Array.from(this._model.metrics || []);
            const keys = new Set(this._metrics.map((metric) => metric.name));
            if (!keys.has('parameters')) {
                let parameters = 0;
                for (const graph of this._model.graphs || []) {
                    const map = new Map((new metrics.Target(graph).metrics || []).map((metric) => [metric.name, metric]));
                    parameters = map.has('parameters') ? parameters + map.get('parameters').value : NaN;
                }
                for (const func of this._model.functions || []) {
                    const map = new Map((new metrics.Target(func).metrics || []).map((metric) => [metric.name, metric]));
                    parameters = map.has('parameters') ? parameters + map.get('parameters').value : NaN;
                }
                if (!Number.isNaN(parameters) && parameters > 0) {
                    this._metrics.push(new metadata.Argument('parameters', parameters, 'attribute'));
                }
            }
        }
        return this._metrics;
    }
};

metrics.Target = class {

    constructor(target) {
        this._target = target;
        this._metrics = null;
    }

    get metrics() {
        if (this._metrics === null) {
            this._metrics = [];
            this._metrics = Array.from(this._target.metrics || []);
            const keys = new Set(this._metrics.map((metrics) => metrics.name));
            if (!keys.has('parameters')) {
                let parameters = 0;
                const initializers = new Set();
                if (this._target && Array.isArray(this._target.nodes)) {
                    for (const node of this._target.nodes) {
                        for (const argument of node.inputs || []) {
                            if (argument && Array.isArray(argument.value)) {
                                for (const value of argument.value) {
                                    if (value && value.initializer) {
                                        initializers.add(value.initializer);
                                    }
                                }
                            }
                        }
                    }
                }
                for (const tensor of initializers) {
                    const shape = tensor && tensor.type && tensor.type.shape && Array.isArray(tensor.type.shape.dimensions) ? tensor.type.shape.dimensions : [];
                    if (!shape.every((dim) => typeof dim === 'number')) {
                        parameters = 0;
                        break;
                    }
                    parameters += shape.reduce((a, b) => a * b, 1);
                }
                if (parameters > 0) {
                    this._metrics.push(new metadata.Argument('parameters', parameters, 'attribute'));
                }
            }
        }
        return this._metrics;
    }
};

metrics.Tensor = class {

    constructor(tensor) {
        this._tensor = tensor;
        this._metrics = null;
    }

    get name() {
        return this._tensor.name || '';
    }

    get metrics() {
        if (this._metrics === null) {
            this._metrics = [];
            this._metrics = Array.from(this._tensor.metrics || []);
            const keys = new Set(this._metrics.map((metrics) => metrics.name));
            const type = this._tensor.type;
            const shape = type.shape.dimensions;
            const size = shape.reduce((a, b) => a * b, 1);
            if (size < 0x800000 &&
                (type.dataType.startsWith('float') || type.dataType.startsWith('bfloat')) &&
                (!keys.has('sparsity') || !keys.has('min') || !keys.has('max') && !keys.has('mean') || !keys.has('max') || !keys.has('std'))) {
                const data = this._tensor.value;
                let zeros = 0;
                let min = null;
                let max = null;
                let sum = 0;
                let count = 0;
                const stack = [data];
                while (stack.length > 0) {
                    const data = stack.pop();
                    if (Array.isArray(data)) {
                        for (const element of data) {
                            stack.push(element);
                        }
                    } else {
                        zeros += data === 0 || data === 0n || data === '';
                        min = Math.min(data, min === null ? data : min);
                        max = Math.max(data, max === null ? data : max);
                        sum += data;
                        count += 1;
                    }
                }
                const mean = sum / count;
                if (!keys.has('sparsity')) {
                    this._metrics.push(new metadata.Argument('min', min, type.dataType));
                }
                if (!keys.has('max')) {
                    this._metrics.push(new metadata.Argument('max', max, type.dataType));
                }
                if (!keys.has('mean')) {
                    this._metrics.push(new metadata.Argument('mean', mean, type.dataType));
                }
                if (!keys.has('std')) {
                    let variance = 0;
                    const stack = [data];
                    while (stack.length > 0) {
                        const data = stack.pop();
                        if (Array.isArray(data)) {
                            for (const element of data) {
                                stack.push(element);
                            }
                        } else {
                            variance += Math.pow(data - mean, 2);
                        }
                    }
                    this._metrics.push(new metadata.Argument('std', Math.sqrt(variance / count)));
                }
                if (!keys.has('sparsity')) {
                    this._metrics.push(new metadata.Argument('sparsity', count > 0 ? zeros / count : 0, 'percentage'));
                }
            }
        }
        return this._metrics;
    }
};

export { metadata, metrics };
