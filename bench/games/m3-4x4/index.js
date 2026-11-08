import { connect } from '../connect.js';

// 4행 × 4열, 3개를 이으면 패배 (c3-4x4의 미제르)
export default connect('m3-4x4', 4, 4, 3, { misere: true });
