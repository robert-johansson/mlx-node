/**
 * Process-local policy adapter for pi's public ModelRegistry.
 *
 * `mlx agent` is an offline/local product, but pi's registry also contains
 * every authenticated built-in cloud provider. CLI `--models mlx/*` only sets
 * the initial selector scope: Tab, `/models`, RPC enumeration, explicit model
 * resolution, and restored sessions still consult the registry's unscoped
 * reads. Filter those reads at their shared boundary before pi constructs its
 * runtime so every path sees only the exact local models this process serves.
 *
 * Keep this adapter isolated: once pi exposes a first-class provider allowlist
 * in `MainOptions`, this file can be replaced by that option without touching
 * the provider or CLI layers.
 */

interface RegistryModel {
  provider: string;
  id: string;
  api: string;
  baseUrl: string;
}

export interface FilterableModelRegistry<TModel extends RegistryModel = RegistryModel> {
  getAll(): TModel[];
  getAvailable(): TModel[];
  find(provider: string, modelId: string): TModel | undefined;
  hasConfiguredAuth(model: TModel): boolean;
}

export interface FilterableModelRegistryConstructor<TModel extends RegistryModel = RegistryModel> {
  prototype: FilterableModelRegistry<TModel>;
}

/**
 * The in-process local providers this agent registers, with the exact
 * baseUrl each one claims. Both are served from local weights with no
 * network and no /login: `mlx` is the v1 native ChatSession provider,
 * `genmlx` the owned-forward one (genmlx-djw6). A model is local only when
 * its provider AND baseUrl match an entry here, so a same-named cloud model
 * can never slip through on id alone.
 */
const LOCAL_PROVIDER_BASE_URLS: ReadonlyMap<string, string> = new Map([
  ['mlx', 'mlx://local'],
  ['genmlx', 'genmlx://local'],
]);

const activePrototypes = new WeakSet<object>();

type RegistryMethodName = keyof FilterableModelRegistry<RegistryModel>;

function requireMethodDescriptor(prototype: object, name: RegistryMethodName): PropertyDescriptor {
  const descriptor = Object.getOwnPropertyDescriptor(prototype, name);
  if (!descriptor || typeof descriptor.value !== 'function' || descriptor.writable !== true) {
    throw new Error(`mlx agent: incompatible pi ModelRegistry.${name}; expected a writable prototype method`);
  }
  return descriptor;
}

/**
 * Install an exact local-model read policy for one `runAgent()` lifetime.
 * Returns an idempotent restore callback.
 */
export function installMlxOnlyModelRegistryFilter<TModel extends RegistryModel>(
  Registry: FilterableModelRegistryConstructor<TModel>,
  modelIds: Iterable<string>,
): () => void {
  const prototype = Registry.prototype;
  if (activePrototypes.has(prototype)) {
    throw new Error('mlx agent: concurrent ModelRegistry filtering in one process is not supported');
  }

  const allowedIds = new Set(modelIds);
  const isAllowed = (model: TModel): boolean =>
    allowedIds.has(model.id) && model.api === 'mlx' && LOCAL_PROVIDER_BASE_URLS.get(model.provider) === model.baseUrl;

  const descriptors = {
    getAll: requireMethodDescriptor(prototype, 'getAll'),
    getAvailable: requireMethodDescriptor(prototype, 'getAvailable'),
    find: requireMethodDescriptor(prototype, 'find'),
    hasConfiguredAuth: requireMethodDescriptor(prototype, 'hasConfiguredAuth'),
  };
  const getAll = descriptors.getAll.value as FilterableModelRegistry<TModel>['getAll'];
  const getAvailable = descriptors.getAvailable.value as FilterableModelRegistry<TModel>['getAvailable'];
  const find = descriptors.find.value as FilterableModelRegistry<TModel>['find'];
  const hasConfiguredAuth = descriptors.hasConfiguredAuth.value as FilterableModelRegistry<TModel>['hasConfiguredAuth'];

  Object.defineProperties(prototype, {
    getAll: {
      ...descriptors.getAll,
      value(this: FilterableModelRegistry<TModel>): TModel[] {
        return getAll.call(this).filter(isAllowed);
      },
    },
    getAvailable: {
      ...descriptors.getAvailable,
      value(this: FilterableModelRegistry<TModel>): TModel[] {
        return getAvailable.call(this).filter(isAllowed);
      },
    },
    find: {
      ...descriptors.find,
      value(this: FilterableModelRegistry<TModel>, provider: string, modelId: string): TModel | undefined {
        if (!LOCAL_PROVIDER_BASE_URLS.has(provider) || !allowedIds.has(modelId)) return undefined;
        const model = find.call(this, provider, modelId);
        return model && isAllowed(model) ? model : undefined;
      },
    },
    hasConfiguredAuth: {
      ...descriptors.hasConfiguredAuth,
      value(this: FilterableModelRegistry<TModel>, model: TModel): boolean {
        return isAllowed(model) && hasConfiguredAuth.call(this, model);
      },
    },
  });

  activePrototypes.add(prototype);
  let restored = false;
  return () => {
    if (restored) return;
    Object.defineProperties(prototype, descriptors);
    activePrototypes.delete(prototype);
    restored = true;
  };
}
