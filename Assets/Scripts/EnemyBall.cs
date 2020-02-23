using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[AddComponentMenu("MyGame/EnemyBall")]
public class EnemyBall : MonoBehaviour
{
	public float m_speed=10;
	public float m_liveTime=1;
	public float m_power=1.0f;
	protected Transform m_transform;
    // Start is called before the first frame update
    void Start()
    {
        m_transform=this.transform;
    }

    // Update is called once per frame
    void Update()
    {
        m_liveTime-=Time.deltaTime;
        if(m_liveTime<=0){
        	Destroy(this.gameObject);
        }
        m_transform.Translate(new Vector3(0,0,m_speed*Time.deltaTime));
    }

    void OnTriggerEnter(Collider other){
        if(other.tag.CompareTo("Player")!=0){
            return;
        }
        Destroy(this.gameObject);
    }
}
