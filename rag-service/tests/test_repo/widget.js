export function makeWidget(name) {
  return { name, kind: 'widget' };
}

export class WidgetStore {
  constructor() {
    this.items = [];
  }
  add(item) {
    this.items.push(item);
  }
}

